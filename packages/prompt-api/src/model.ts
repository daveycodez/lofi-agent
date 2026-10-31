/**
 * The browser's built-in model: whether this browser has one, which one it
 * is, and its download. The browser downloads the model once for all sites
 * (Chrome keeps it in its profile) and loads it when a session is created.
 */
import { createStore } from "./store.js"
import type { Store } from "./store.js"
import { languageModel } from "./types.js"
import type {
  Availability,
  DownloadProgressEvent,
  LanguageModelCoreOptions,
  LanguageModelCreateOptions,
  LanguageModelSession,
  SamplingMode
} from "./types.js"

export type { Availability, SamplingMode } from "./types.js"

/** Same shape as the other adapters' loading status. */
export type ModelStatus =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 of the browser's download of the model. */
      progress: number
      /** True when the model is already on the device, not downloading. */
      cached: boolean
      /** Bytes downloaded so far: the Prompt API never says, so always unset. */
      loadedBytes?: number
      /** Bytes of the whole download: always unset, as `loadedBytes`. */
      totalBytes?: number
    }
  | { state: "ready" }
  | { state: "error"; message: string }

/** Kinds of input besides text a session can be told to expect. */
export type MediaInput = "image" | "audio"

/** Languages the model is told to expect, in and out, by default. */
export const DEFAULT_LANGUAGES = ["en"] as const

export const NO_PROMPT_API =
  "This browser has no built-in language model (the Prompt API). It's in Chrome and Edge on desktop."
export const MODEL_UNAVAILABLE =
  "This browser's built-in model can't run on this device (not enough memory, GPU or disk space, or turned off by policy)."

/** An error whose message is already written for the user. */
export class PromptApiError extends Error {
  override readonly name = "PromptApiError"
}

/**
 * What every session expects: text in and out in the languages given (Chrome
 * checks its safety filters for the output language), images or audio in if
 * the conversation has them, and tool calls if the browser runs tools.
 */
export function baseOptions(
  languages: ReadonlyArray<string> = DEFAULT_LANGUAGES,
  {
    media = [],
    nativeTools = false
  }: { media?: ReadonlyArray<MediaInput>; nativeTools?: boolean } = {}
): LanguageModelCreateOptions {
  return {
    expectedInputs: [
      { type: "text", languages: [...languages] },
      ...media.map((type) => ({ type })),
      ...(nativeTools
        ? [{ type: "tool-call" as const }, { type: "tool-response" as const }]
        : [])
    ],
    expectedOutputs: [
      { type: "text", languages: [...languages] },
      ...(nativeTools ? [{ type: "tool-call" as const }] : [])
    ]
  }
}

/**
 * Whether this browser can run its built-in model: "available" (on the
 * device), "downloadable" or "downloading" (it downloads when a session is
 * created), or "unavailable" (no Prompt API, or the device can't run it).
 * Give it what the sessions will use: the `languages`, any `samplingMode`,
 * and `inputs` besides text (images need more of the device, audio a GPU).
 */
export async function promptApiAvailability(
  languages?: ReadonlyArray<string>,
  options: {
    samplingMode?: SamplingMode
    inputs?: ReadonlyArray<MediaInput>
  } = {}
): Promise<Availability> {
  const api = languageModel()
  if (!api) return "unavailable"
  try {
    return await api.availability({
      ...baseOptions(languages, { media: options.inputs }),
      ...(options.samplingMode && { samplingMode: options.samplingMode })
    })
  } catch {
    return "unavailable"
  }
}

const status = createStore<ModelStatus>({ state: "idle" })

/** Loading status of the built-in model: its download, then ready. */
export const promptApiStatus = (): Store<ModelStatus> => status

let contextWindow: number | undefined
/** Records a session's context window, the same for every session. */
export function noteContextWindow(session: LanguageModelSession) {
  if (session.contextWindow > 0) contextWindow = session.contextWindow
}

/** What a failed create() means for the user. */
export function createErrorText(error: unknown) {
  const name = error instanceof DOMException ? error.name : ""
  if (name === "NotAllowedError") {
    return "The browser wants a click on the page before it downloads its built-in model. Send the message again."
  }
  if (name === "NotSupportedError") return MODEL_UNAVAILABLE
  if (name === "QuotaExceededError") {
    return "Not enough disk space for the browser's built-in model."
  }
  if (name === "NetworkError") {
    return "The browser couldn't download its built-in model. Check the connection and try again."
  }
  return `The built-in model couldn't start: ${error instanceof Error ? error.message : String(error)}`
}

const asksForMedia = (options: LanguageModelCreateOptions) =>
  options.expectedInputs?.some((e) => e.type === "image" || e.type === "audio")

/**
 * A new session, with `promptApiStatus()` kept up to date: until a session
 * has started, it checks `availability()` with the same options and shows
 * the download; after, only a download shows (if the browser removed the
 * model to free space, say). `signal` stops the wait, not the session made:
 * a session given a request's signal would be destroyed by a later abort of
 * that request, after it was kept for the next one.
 */
export async function createLanguageModel(
  options: LanguageModelCreateOptions,
  signal?: AbortSignal
): Promise<LanguageModelSession> {
  const api = languageModel()
  if (!api) {
    status.set({ state: "error", message: NO_PROMPT_API })
    throw new PromptApiError(NO_PROMPT_API)
  }
  signal?.throwIfAborted()
  const {
    expectedInputs,
    expectedOutputs,
    tools,
    samplingMode,
    temperature,
    topK
  } = options
  const core: LanguageModelCoreOptions = {
    expectedInputs,
    expectedOutputs,
    ...(tools && { tools }),
    ...(samplingMode !== undefined && { samplingMode }),
    ...(temperature !== undefined && { temperature }),
    ...(topK !== undefined && { topK })
  }
  const starting = status.get().state !== "ready"
  if (starting) {
    const availability = await api.availability(core).catch(() => "unavailable")
    if (availability === "unavailable") {
      if (asksForMedia(options)) {
        throw new PromptApiError(
          "This browser's built-in model can't read images or audio on this device."
        )
      }
      status.set({ state: "error", message: MODEL_UNAVAILABLE })
      throw new PromptApiError(MODEL_UNAVAILABLE)
    }
    if (status.get().state !== "loading") {
      const downloaded = availability === "available"
      status.set({
        state: "loading",
        progress: downloaded ? 1 : 0,
        cached: downloaded
      })
    }
  }

  const controller = new AbortController()
  const onAbort = () => controller.abort(signal?.reason)
  signal?.addEventListener("abort", onAbort, { once: true })
  try {
    const session = await api.create({
      ...options,
      signal: controller.signal,
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) => {
          // Every create() reports 0 and 1, downloading or not.
          const { loaded } = event as DownloadProgressEvent
          if (loaded > 0 && loaded < 1) {
            status.set({ state: "loading", progress: loaded, cached: false })
          }
        })
      }
    })
    noteContextWindow(session)
    if (status.get().state !== "ready") status.set({ state: "ready" })
    return session
  } catch (error) {
    if (signal?.aborted) {
      if (starting && status.get().state === "loading") {
        status.set({ state: "idle" })
      }
      throw error
    }
    const message = createErrorText(error)
    if (starting && !asksForMedia(options)) {
      status.set({ state: "error", message })
    }
    throw new PromptApiError(message, { cause: error })
  } finally {
    signal?.removeEventListener("abort", onAbort)
  }
}

/**
 * Tokens a session holds, prompt and replies together (9,216 for Gemini
 * Nano in Chrome 156). Creates and frees a session to read it if none was
 * made yet; undefined while the model isn't on the device.
 */
export async function promptApiContextWindow(
  languages?: ReadonlyArray<string>
): Promise<number | undefined> {
  if (contextWindow) return contextWindow
  if ((await promptApiAvailability(languages)) !== "available") return undefined
  const session = await createLanguageModel(baseOptions(languages))
  session.destroy()
  return contextWindow
}

let loading: Promise<void> | null = null

/**
 * Downloads the model if it isn't on the device, with its progress in
 * `promptApiStatus()`, and loads it. `chat()` doesn't need it (its first
 * session does the same), but the browser asks for a click on the page
 * before a download: call it from the click that turns the model on. The
 * download goes on if `signal` aborts; only the wait stops.
 */
export function loadPromptApiModel(
  options: { languages?: ReadonlyArray<string>; signal?: AbortSignal } = {}
): Promise<void> {
  if (status.get().state === "ready") return Promise.resolve()
  loading ??= createLanguageModel(baseOptions(options.languages))
    .then((session) => session.destroy())
    .finally(() => {
      loading = null
    })
  const { signal } = options
  if (!signal) return loading
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => reject(signal.reason)
    if (signal.aborted) return onAbort()
    signal.addEventListener("abort", onAbort, { once: true })
    ;(loading ?? Promise.resolve()).then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort)
    })
  })
}

export type PromptApiModelInfo = {
  /** The model, where the browser tells it: by which browser this is. */
  id: "gemini-nano" | "phi-4-mini" | "unknown"
  /** "Gemini Nano", "Phi-4-mini", or "Built-in model". */
  name: string
  /** "Chrome", "Edge", or the browser's brand; null if it can't be told. */
  browser: string | null
}

type NavigatorBrands = {
  userAgentData?: { brands?: Array<{ brand: string }> }
}

/**
 * Which model this browser provides. The Prompt API doesn't say, but each
 * browser has its own: Gemini Nano in Chrome, Phi-4-mini in Edge. Read from
 * the browser's brand (User-Agent Client Hints), else its user agent.
 */
export function promptApiModel(): PromptApiModelInfo {
  if (typeof navigator === "undefined") {
    return { id: "unknown", name: "Built-in model", browser: null }
  }
  const brands = ((navigator as NavigatorBrands).userAgentData?.brands ?? [])
    .map((b) => b.brand)
    .filter((b) => !/not.?a.?brand/i.test(b))
  const ua = navigator.userAgent
  if (brands.includes("Microsoft Edge") || /\bEdg\//.test(ua)) {
    return { id: "phi-4-mini", name: "Phi-4-mini", browser: "Edge" }
  }
  if (
    brands.includes("Google Chrome") ||
    (brands.length === 0 && /\bChrome\//.test(ua) && !/\bOPR\//.test(ua))
  ) {
    return { id: "gemini-nano", name: "Gemini Nano", browser: "Chrome" }
  }
  const brand = brands.find((b) => b !== "Chromium") ?? brands[0] ?? null
  return { id: "unknown", name: "Built-in model", browser: brand }
}
