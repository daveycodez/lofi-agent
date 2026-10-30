/**
 * The page's side of the Transformers.js worker (worker.ts), which loads and
 * runs the model off the main thread so the page stays smooth. One worker,
 * and one model in it at a time: loading another frees the previous one.
 */
import { createStore } from "./store.js"
import type { Store } from "./store.js"
import type {
  ChatTurn,
  RunTimings,
  Sampling,
  ToolSpec,
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
    }
  | { state: "ready" }
  | { state: "error"; message: string }

const statuses = new Map<string, Store<ModelStatus>>()

/** Loading status of a model, by Hugging Face id. */
export function modelStatus(modelId: string): Store<ModelStatus> {
  let status = statuses.get(modelId)
  if (!status) {
    status = createStore<ModelStatus>({ state: "idle" })
    statuses.set(modelId, status)
  }
  return status
}

/** Listeners for each running generation, by request id. */
type Generation = {
  onToken: (text: string) => void
  onStats: (stats: RunTimings) => void
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
    modelStatus(data.modelId).set({
      state: "loading",
      progress: data.progress,
      cached: data.cached
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
  } else if (data.type === "stats") {
    generations.get(data.id)?.onStats(data.stats)
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
 * Only pages can ask, not workers.
 */
let askedToKeep = false
function keepDownloads() {
  if (askedToKeep || typeof navigator === "undefined") return
  askedToKeep = true
  navigator.storage
    ?.persist?.()
    .then((kept) => {
      if (debug) console.info(`[download] storage kept: ${kept}`)
    })
    .catch(() => undefined)
}

/** Loads a model (downloading it the first time), replacing any other. */
export function loadModel(modelId: string): Promise<void> {
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
  keepDownloads()
  // The worker reports whether this is a download or a cached load.
  modelStatus(modelId).set({ state: "loading", progress: 0, cached: true })
  post({ type: "load", modelId })
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
  thinking: boolean
  sampling: Sampling
  signal?: AbortSignal
  /** The run's counts so far, a few times a second while it runs. */
  onStats?: (stats: RunTimings) => void
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
    onStats: (stats) => request.onStats?.(stats),
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
    thinking: request.thinking,
    sampling: request.sampling
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
