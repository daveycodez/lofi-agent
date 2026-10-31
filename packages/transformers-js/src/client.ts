/**
 * The page's side of the Transformers.js worker (worker.ts), which loads and
 * runs the model off the main thread so the page stays smooth. One worker,
 * and one model in it at a time: loading another frees the previous one.
 */
import { modelMeta, runtimeModelId } from "./model-meta.js"
import { createStore } from "./store.js"
import type { Store } from "./store.js"
import type {
  ChatTurn,
  RunTimings,
  Sampling,
  ToolSpec,
  WeightsDtype,
  WorkerRequest,
  WorkerResponse
} from "./protocol.js"

export type ModelStatus =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 across the model's files. */
      progress: number
      /** True when the files come from the browser's storage, not the network. */
      cached: boolean
      /** Bytes downloaded so far, while downloading. */
      loadedBytes?: number
      /** Bytes of all the model's files, while downloading. */
      totalBytes?: number
    }
  | { state: "ready" }
  | { state: "error"; message: string }

const statuses = new Map<string, Store<ModelStatus>>()

/**
 * Loading status of a model, by its name: a listed model or any Hugging Face
 * id.
 */
export function modelStatus(model: string): Store<ModelStatus> {
  const id = runtimeModelId(model)
  let status = statuses.get(id)
  if (!status) {
    status = createStore<ModelStatus>({ state: "idle" })
    statuses.set(id, status)
  }
  return status
}

/** Listeners for each running generation, by request id. */
type Generation = {
  onToken: (text: string) => void
  onEnd: (error?: string, stats?: RunTimings) => void
}
const generations = new Map<string, Generation>()

type Pending = {
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
}
/** The model the worker holds (or is loading), and its load promise. */
let loaded: { modelId: string; load: Pending } | null = null

let worker: Worker | null = null
let debug = false

/**
 * Turns on the worker's diagnostics (memory and cache logs, error stacks in
 * the console). Takes effect if called before the worker starts.
 */
export function setDebug(on: boolean) {
  debug = on
}

function handleMessage({ data }: MessageEvent<WorkerResponse>) {
  if (data.type === "progress") {
    if (!data.cached) keepDownloads()
    modelStatus(data.modelId).set({
      state: "loading",
      progress: data.progress,
      cached: data.cached,
      ...(data.totalBytes !== undefined && {
        loadedBytes: data.loadedBytes ?? 0,
        totalBytes: data.totalBytes
      })
    })
  } else if (data.type === "ready") {
    modelStatus(data.modelId).set({ state: "ready" })
    if (loaded?.modelId === data.modelId) loaded.load.resolve()
  } else if (data.type === "load-error") {
    modelStatus(data.modelId).set({ state: "error", message: data.message })
    if (loaded?.modelId === data.modelId) {
      loaded.load.reject(new Error(data.message))
      loaded = null
    }
  } else if (data.type === "token") {
    generations.get(data.id)?.onToken(data.text)
  } else if (data.type === "log") {
    // A worker's own console output doesn't reach a dev server's terminal.
    console.info(data.text)
  } else {
    const generation = generations.get(data.id)
    generations.delete(data.id)
    if (data.type === "error") generation?.onEnd(data.message)
    else generation?.onEnd(undefined, data.stats)
  }
}

function post(message: WorkerRequest) {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
      // The worker reads its name to turn on diagnostics before any message.
      name: debug ? "transformers-js:debug" : "transformers-js"
    })
    worker.onmessage = handleMessage
  }
  worker.postMessage(message)
}

/**
 * Asks the browser to keep this site's storage, so a downloaded model isn't
 * deleted to free space or, in Safari, after 7 days without a visit (which
 * means downloading it again). Browsers may say no; it's only a request.
 * Only pages can ask, not workers. Called when a download starts, not for
 * cached loads. Skipped in Firefox, which shows a permission prompt for it;
 * there a model is still kept across reloads, unless the disk runs low.
 */
let askedToKeep = false
function keepDownloads() {
  if (askedToKeep || typeof navigator === "undefined") return
  askedToKeep = true
  if (/firefox/i.test(navigator.userAgent)) return
  navigator.storage
    ?.persist?.()
    .then((kept) => {
      if (debug) console.info(`[download] storage kept: ${kept}`)
    })
    .catch(() => undefined)
}

let webGpu: Promise<boolean> | undefined

/**
 * Whether models here run on WebGPU: the browser has it and gives a GPU
 * adapter. Where it doesn't, they run on the CPU through WebAssembly, which
 * needs a model with q4 weights (see `runsWithoutWebGpu`). The worker makes
 * the same check when it loads a model.
 */
export function hasWebGpu(): Promise<boolean> {
  webGpu ??= (async () => {
    const nav =
      typeof navigator === "undefined"
        ? undefined
        : (navigator as Navigator & {
            gpu?: { requestAdapter: () => Promise<unknown> }
          })
    if (!nav?.gpu) return false
    try {
      return (await nav.gpu.requestAdapter()) !== null
    } catch {
      return false
    }
  })()
  return webGpu
}

/**
 * Loads a model (downloading it the first time), replacing any other: a
 * listed model at its pinned revision, or any Hugging Face id. Without
 * WebGPU the worker loads its q4 weights on the CPU instead of `dtype`.
 */
export function loadModel(
  model: string,
  dtype: WeightsDtype = modelMeta(model)?.dtype ?? "q4f16"
): Promise<void> {
  const modelId = runtimeModelId(model)
  if (loaded?.modelId === modelId) return loaded.load.promise
  if (loaded) {
    // The worker frees the previous model before loading this one.
    modelStatus(loaded.modelId).set({ state: "idle" })
    loaded.load.reject(new Error("Switched to another model."))
  }
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((res, rej) => {
    resolve = res
    reject = rej
  })
  promise.catch(() => undefined)
  loaded = { modelId, load: { promise, resolve, reject } }
  // The worker reports whether this is a download or a cached load.
  modelStatus(modelId).set({ state: "loading", progress: 0, cached: true })
  post({ type: "load", modelId, dtype })
  return promise
}

/** What a finished generation left: its error, if it failed, and its counts. */
export type GenerationEnd = { failure?: string; stats?: RunTimings }

/**
 * Runs one generation with the loaded model, yielding its text as it comes
 * (thinking and tool-call markup included, as the model writes them).
 */
export async function* generate(request: {
  id: string
  chat: Array<ChatTurn>
  tools: Array<ToolSpec>
  thinking?: boolean
  sampling: Sampling
  maxTokens?: number
  contextLength: number
  thinkingOpen: string
  signal?: AbortSignal
}): AsyncGenerator<string, GenerationEnd> {
  const { id, signal } = request
  // Bridge the worker's messages to this async generator.
  const queue: Array<string> = []
  let wake: (() => void) | null = null
  let finished = false
  let end: GenerationEnd = {}
  generations.set(id, {
    onToken: (text) => {
      queue.push(text)
      wake?.()
    },
    onEnd: (failure, stats) => {
      end = { failure, stats }
      finished = true
      wake?.()
    }
  })
  const onAbort = () => post({ type: "stop", id })
  signal?.addEventListener("abort", onAbort)
  post({
    type: "generate",
    id,
    chat: request.chat,
    tools: request.tools,
    ...(request.thinking !== undefined && { thinking: request.thinking }),
    sampling: request.sampling,
    ...(request.maxTokens !== undefined && { maxTokens: request.maxTokens }),
    contextLength: request.contextLength,
    thinkingOpen: request.thinkingOpen
  })
  try {
    while (!finished || queue.length > 0) {
      const text = queue.shift()
      if (text !== undefined) {
        yield text
        continue
      }
      await new Promise<void>((resolve) => {
        wake = resolve
      })
      wake = null
    }
  } finally {
    signal?.removeEventListener("abort", onAbort)
  }
  return end
}
