/**
 * The page's side of wllama: llama.cpp compiled to WebAssembly, which runs
 * in wllama's own Web Worker. One model at a time: loading another frees
 * the previous one. Models download once to the Origin Private File
 * System, and load from there afterwards, offline included.
 */
import { LoggerWithoutDebug, Wllama } from "@wllama/wllama/esm/index.js"
import type {
  ChatCompletionChunk,
  ChatCompletionParams,
  ChatCompletionResponse
} from "@wllama/wllama/esm/index.js"

import { modelMeta, modelUrl } from "./model-meta.js"
import type { WllamaTextModel } from "./model-meta.js"
import { createStore } from "./store.js"
import type { Store } from "./store.js"

export type ModelStatus =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 across the model's file. */
      progress: number
      /** True when the file comes from the browser's storage, not the network. */
      cached: boolean
      /** Bytes downloaded so far, while downloading. */
      loadedBytes?: number
      /** Bytes of the model's file, while downloading. */
      totalBytes?: number
    }
  | { state: "ready" }
  | { state: "error"; message: string }

const statuses = new Map<string, Store<ModelStatus>>()

/** Loading status of a model, by its name. */
export function modelStatus(model: WllamaTextModel): Store<ModelStatus> {
  let status = statuses.get(model)
  if (!status) {
    status = createStore<ModelStatus>({ state: "idle" })
    statuses.set(model, status)
  }
  return status
}

/** Options that apply to the runtime, so to every model. */
export type RuntimeOptions = {
  /**
   * Where wllama's WebAssembly file is served. By default the bundler
   * emits it from `@wllama/wllama` (see the README).
   */
  wasmUrl?: string
  /**
   * Where wllama's compat build is served: its WebAssembly file and its
   * worker script. Browsers without JSPI or 64-bit WebAssembly memory
   * (Safari) run it instead of the main build, which doesn't run there. By
   * default the bundler emits both from `@wllama/wllama-compat`.
   */
  compat?: { wasm: string; worker: string }
  /** CPU threads. wllama's default otherwise: half the cores. */
  threads?: number
  /** wllama's and llama.cpp's logs, in the console. */
  debug?: boolean
}

let options: RuntimeOptions = {}

/** Sets runtime options; they apply from the next model load. */
export function configureRuntime(next: RuntimeOptions) {
  options = { ...options, ...next }
}

const defaultWasmUrl = () =>
  new URL("@wllama/wllama/esm/wasm/wllama.wasm", import.meta.url).href

const defaultCompat = () => ({
  wasm: new URL("@wllama/wllama-compat/wasm/wllama.wasm", import.meta.url)
    .href,
  worker: new URL("@wllama/wllama-compat/wasm/wllama.js", import.meta.url).href
})

/** A WebAssembly module with only a 64-bit memory. */
const MEMORY64_MODULE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 1
])

/**
 * Whether wllama's compat build has to run here: without JSPI, or without
 * 64-bit WebAssembly memory, checked by compiling a module that has one.
 * (Safari accepts `new WebAssembly.Memory({ address: "i64" })`, wllama's own
 * check, but can't compile the main build.) Not in Firefox, which runs the
 * main build without JSPI.
 */
function needsCompat() {
  if (/Firefox\//.test(navigator.userAgent)) return false
  return (
    !("Suspending" in WebAssembly) || !WebAssembly.validate(MEMORY64_MODULE)
  )
}

/**
 * Runs `wllama` on its compat build. wllama only uses one where its own
 * checks say so, and they pass in Safari, which then fails to compile the
 * main build (and the load never ends), so its worker's resources are set
 * here too.
 */
function useCompat(
  wllama: Wllama,
  compat: { wasm: string; worker: { code: string } }
) {
  wllama.setCompat(compat)
  const target = wllama as unknown as { getWorkerResources: () => object }
  const resources = target.getWorkerResources.bind(wllama)
  target.getWorkerResources = () => ({
    ...resources(),
    wasmPath: compat.wasm,
    jsPath: compat.worker,
    compat: true
  })
}

let compatCode: Promise<string> | undefined

/**
 * wllama's compat build, served with the site rather than from wllama's
 * CDN, so it works offline too. wllama wraps the worker script in a
 * function, so it's passed as code that ends in a newline: a bundler's
 * dev server may end it in a `//# sourceMappingURL` comment, which would
 * otherwise comment out the end of that function.
 */
async function compatBuild() {
  const { wasm, worker } = options.compat ?? defaultCompat()
  compatCode ??= fetch(worker).then(async (response) => {
    if (!response.ok) {
      throw new Error(
        `Couldn't load wllama's compat build (${response.status}).`
      )
    }
    return `${await response.text()}\n`
  })
  compatCode.catch(() => {
    compatCode = undefined
  })
  return { wasm, worker: { code: await compatCode } }
}

/** The model wllama holds (or is loading), its load, and a way to stop it. */
let loaded: {
  model: WllamaTextModel
  wllama: Wllama
  promise: Promise<Wllama>
  abort: AbortController
} | null = null

let askedToKeep = false
/**
 * Asks the browser to keep this site's storage, so a downloaded model isn't
 * deleted to free space or, in Safari, after 7 days without a visit.
 * Browsers may say no; it's only a request.
 */
function keepDownloads() {
  if (askedToKeep || typeof navigator === "undefined") return
  askedToKeep = true
  navigator.storage?.persist?.().catch(() => undefined)
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

async function startLoad(
  model: WllamaTextModel,
  wllama: Wllama,
  signal: AbortSignal
) {
  const status = modelStatus(model)
  const url = modelUrl(model)
  const meta = modelMeta(model)
  if (needsCompat()) useCompat(wllama, await compatBuild())
  // A model already in storage loads from it, without the network.
  const saved = await wllama.modelManager.getModels()
  const cached = saved.some((m) => m.url === url)
  status.set({ state: "loading", progress: 0, cached })
  await wllama.loadModelFromUrl(url, {
    // wllama's default is 1,024 tokens, less than a system prompt with tools.
    n_ctx: meta.context,
    // On the CPU: wllama would otherwise use WebGPU where the browser has it.
    n_gpu_layers: 0,
    ...(options.threads && { n_threads: options.threads }),
    signal,
    progressCallback: ({ loaded: bytes, total }) => {
      if (cached) return
      const totalBytes = total || meta.bytes
      status.set({
        state: "loading",
        progress: Math.min(bytes / totalBytes, 1),
        cached: false,
        loadedBytes: bytes,
        totalBytes
      })
    }
  })
  return wllama
}

/**
 * Loads a model (downloading it the first time), replacing any other. The
 * promise settles when it's ready to generate.
 */
export function loadModel(model: WllamaTextModel): Promise<Wllama> {
  if (loaded?.model === model) return loaded.promise
  const previous = loaded
  if (previous) modelStatus(previous.model).set({ state: "idle" })
  keepDownloads()
  const wllama = new Wllama(
    { default: options.wasmUrl ?? defaultWasmUrl() },
    {
      logger: options.debug ? console : LoggerWithoutDebug,
      suppressNativeLog: !options.debug
    }
  )
  const status = modelStatus(model)
  status.set({ state: "loading", progress: 0, cached: true })
  const abort = new AbortController()
  const promise = (async () => {
    if (previous) {
      // Stop its download, if it's still downloading, and free it.
      previous.abort.abort()
      await previous.promise.catch(() => undefined)
      await previous.wllama.exit().catch(() => undefined)
    }
    return startLoad(model, wllama, abort.signal)
  })()
  const current = { model, wllama, promise, abort }
  loaded = current
  promise.then(
    () => {
      if (loaded === current) status.set({ state: "ready" })
    },
    (error: unknown) => {
      if (loaded !== current) return
      loaded = null
      status.set({ state: "error", message: errorText(error) })
      void wllama.exit().catch(() => undefined)
    }
  )
  promise.catch(() => undefined)
  return promise
}

const count = new Intl.NumberFormat("en-US")

/** llama.cpp's error for a prompt past the context limit, said plainly. */
export function contextError(error: unknown, context: number) {
  const tokens = /request \((\d+) tokens\) exceeds the available context/.exec(
    errorText(error)
  )?.[1]
  if (!tokens) return error
  return new Error(
    `This session is too long for the model: ${count.format(Number(tokens))} tokens, and it can read ${count.format(context)} at most, reply included. Start a new session to keep going.`
  )
}

/** A chat completion request, as llama.cpp's OpenAI-compatible API takes it. */
export type ChatRequest = Omit<ChatCompletionParams, "stream" | "abortSignal">

/** Streams a chat completion from `model`, loading it first. */
export async function streamChat(
  model: WllamaTextModel,
  request: ChatRequest,
  signal?: AbortSignal
): Promise<AsyncIterable<ChatCompletionChunk>> {
  const wllama = await loadModel(model)
  return wllama.createChatCompletion({
    ...request,
    stream: true,
    ...(signal && { abortSignal: signal })
  })
}

/** A whole chat completion from `model`, loading it first. */
export async function completeChat(
  model: WllamaTextModel,
  request: ChatRequest,
  signal?: AbortSignal
): Promise<ChatCompletionResponse> {
  const wllama = await loadModel(model)
  try {
    return await wllama.createChatCompletion({
      ...request,
      stream: false,
      ...(signal && { abortSignal: signal })
    })
  } catch (error) {
    throw contextError(error, modelMeta(model).context)
  }
}
