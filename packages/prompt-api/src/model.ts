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
  LanguageModelCreateOptions,
  LanguageModelSession
} from "./types.js"

export type { Availability } from "./types.js"

/** Same shape as the other adapters' loading status. */
export type ModelStatus =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 of the browser's download of the model. */
      progress: number
      /** True when the model is already on the device, not downloading. */
      cached: boolean
      /** Bytes downloaded so far, when the browser says (Chrome doesn't). */
      loadedBytes?: number
      /** Bytes of the whole download, when the browser says. */
      totalBytes?: number
    }
  | { state: "ready" }
  | { state: "error"; message: string }

/** Languages the model is told to expect, in and out, by default. */
export const DEFAULT_LANGUAGES = ["en"] as const

export const NO_PROMPT_API =
  "This browser has no built-in language model (the Prompt API). It's in Chrome and Edge on desktop."
export const MODEL_UNAVAILABLE =
  "This browser's built-in model can't run on this device (not enough memory, GPU or disk space, or turned off by policy)."

/**
 * Options every session is created with: text in and out, in the languages
 * given. Chrome asks pages to name an output language; it checks its safety
 * filters for that language.
 */
export function baseOptions(
  languages: ReadonlyArray<string> = DEFAULT_LANGUAGES
): LanguageModelCreateOptions {
  return {
    expectedInputs: [{ type: "text", languages: [...languages] }],
    expectedOutputs: [{ type: "text", languages: [...languages] }]
  }
}

/**
 * Whether this browser can run its built-in model: "available" (on the
 * device), "downloadable" or "downloading" (it downloads when a session is
 * created), or "unavailable" (no Prompt API, or the device can't run it).
 */
export async function promptApiAvailability(
  languages?: ReadonlyArray<string>
): Promise<Availability> {
  const api = languageModel()
  if (!api) return "unavailable"
  try {
    return await api.availability(baseOptions(languages))
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
  const session = await languageModel()?.create(baseOptions(languages))
  if (session) {
    noteContextWindow(session)
    session.destroy()
  }
  return contextWindow
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
  return `The built-in model couldn't start: ${error instanceof Error ? error.message : String(error)}`
}

let loading: Promise<void> | null = null

/**
 * Makes sure the model is on the device and loaded: downloads it if it
 * isn't, with its progress in `promptApiStatus()`, then loads it. Resolves
 * once a session can start. The download goes on if `signal` aborts; only
 * the wait stops.
 */
export function loadPromptApiModel(
  options: { languages?: ReadonlyArray<string>; signal?: AbortSignal } = {}
): Promise<void> {
  const current = status.get()
  if (current.state === "ready") return Promise.resolve()
  loading ??= download(options.languages).finally(() => {
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

async function download(languages?: ReadonlyArray<string>) {
  const api = languageModel()
  if (!api) {
    status.set({ state: "error", message: NO_PROMPT_API })
    throw new Error(NO_PROMPT_API)
  }
  const availability = await promptApiAvailability(languages)
  if (availability === "unavailable") {
    status.set({ state: "error", message: MODEL_UNAVAILABLE })
    throw new Error(MODEL_UNAVAILABLE)
  }
  const downloaded = availability === "available"
  status.set({
    state: "loading",
    progress: downloaded ? 1 : 0,
    cached: downloaded
  })
  try {
    // Creating a session starts the download, or loads the model into
    // memory if it's on the device (6 seconds in Chrome, the first time
    // after it starts). The session is only for that: the browser keeps
    // the model loaded for the sessions that follow.
    const session = await api.create({
      ...baseOptions(languages),
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) => {
          const { loaded } = event as DownloadProgressEvent
          status.set({ state: "loading", progress: loaded, cached: false })
        })
      }
    })
    noteContextWindow(session)
    session.destroy()
    status.set({ state: "ready" })
  } catch (error) {
    const message = createErrorText(error)
    status.set({ state: "error", message })
    throw new Error(message)
  }
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
