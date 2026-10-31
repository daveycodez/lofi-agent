/**
 * The page's side of wllama: llama.cpp compiled to WebAssembly, which runs
 * in wllama's own Web Worker. One model at a time: loading another frees
 * the previous one. Models download once to the Origin Private File
 * System, and load from there afterwards, offline included.
 */
import { BUILD_ID } from "../vendor/wllama/build-id.js"
import { LoggerWithoutDebug, Wllama } from "../vendor/wllama/index.js"
import type {
  ChatCompletionChunk,
  ChatCompletionParams,
  ChatCompletionResponse
} from "../vendor/wllama/index.js"

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
   * Where wllama's main WebAssembly file is served, for every browser that
   * runs the main build. By default the bundler emits this package's
   * builds (`wasm/`, see the README and scripts/build-wasm.sh).
   */
  wasmUrl?: string
  /**
   * Where this package's relaxed SIMD build is served
   * (`wasm/wllama-relaxed.wasm`), or `false` to use the main build without
   * it. Browsers with relaxed SIMD (Chrome, Firefox) use it; by default the
   * bundler emits it from this package.
   */
  relaxedSimdWasmUrl?: string | false
  /**
   * Where the compat build is served: its WebAssembly file and its worker
   * script. Browsers without JSPI or 64-bit WebAssembly memory (Safari) run
   * it instead of the main build, which doesn't run there. By default the
   * bundler emits both from this package (`wasm/wllama-compat.*`).
   */
  compat?: { wasm: string; worker: string }
  /** CPU threads. wllama's default otherwise: half the cores. */
  threads?: number
  /**
   * `false` to run on the CPU only. By default, wllama's: on WebGPU where
   * the browser has it (Chrome, Edge, Safari), on the CPU elsewhere
   * (Firefox, which has no JSPI, and browsers without WebGPU).
   */
  gpu?: boolean
  /** wllama's and llama.cpp's logs, in the console. */
  debug?: boolean
}

let options: RuntimeOptions = {}

/** Sets runtime options; they apply from the next model load. */
export function configureRuntime(next: RuntimeOptions) {
  options = { ...options, ...next }
}

// This package's builds of wllama (scripts/build-wasm.sh): llama.cpp new
// enough for LFM3, with its WebAssembly SIMD kernels, which wllama's own
// builds leave out. wllama's JavaScript (vendor/wllama) comes from the same
// build, with slot state save and restore added.

/**
 * The full URL of a file of this package's `wasm/`, from its URL next to
 * this module. Turbopack (Next.js) makes that URL a path, which the compat
 * worker can't load. Vite's dev server pre-bundles the package into
 * node_modules/.vite/deps, which moves these URLs to node_modules/.vite/wasm,
 * where it serves no files; it does serve the package's own, so they're used
 * instead.
 */
function packageFile(url: URL) {
  const file = new URL(url.href, location.href)
  if (!file.pathname.includes("/node_modules/.vite/")) return file.href
  const name = file.pathname.slice(file.pathname.lastIndexOf("/") + 1)
  return new URL(`/node_modules/@tanstack-ai-browser/wllama/wasm/${name}`, file)
    .href
}

/** The main build. */
const defaultWasmUrl = () =>
  packageFile(new URL("../wasm/wllama.wasm", import.meta.url))

/**
 * The main build with relaxed SIMD dot products as well: about twice as
 * fast at reading prompts as wllama's own build.
 */
const relaxedWasmUrl = () =>
  packageFile(new URL("../wasm/wllama-relaxed.wasm", import.meta.url))

/** A WebAssembly module with one relaxed SIMD instruction (i32x4.relaxed_dot_i8x16_i7x16_add_s). */
const RELAXED_SIMD_MODULE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 123, 1, 123, 3, 2, 1, 0, 10, 13,
  1, 11, 0, 32, 0, 32, 0, 32, 0, 253, 147, 2, 11
])

function hasRelaxedSimd() {
  try {
    return WebAssembly.validate(RELAXED_SIMD_MODULE)
  } catch {
    return false
  }
}

/** The main build to load: the faster one where the browser can run it. */
function mainWasmUrl() {
  if (options.wasmUrl) return options.wasmUrl
  if (options.relaxedSimdWasmUrl !== false && hasRelaxedSimd()) {
    return options.relaxedSimdWasmUrl ?? relaxedWasmUrl()
  }
  return defaultWasmUrl()
}

/**
 * The compat build, with the glue from the same build. The glue ships as
 * `.js.txt` so bundlers emit it as is: `next build --webpack` minifies `.js`
 * files emitted this way, which breaks it.
 */
const defaultCompat = () => ({
  wasm: packageFile(new URL("../wasm/wllama-compat.wasm", import.meta.url)),
  worker: packageFile(new URL("../wasm/wllama-compat.js.txt", import.meta.url))
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
 * The compat build, served with the site rather than from wllama's CDN,
 * so it works offline too. wllama wraps the worker script in a
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

// The model's state after a chat (what it has read, as llama.cpp's slot
// save file), kept in the Origin Private File System so the first message
// after a reload doesn't read the system prompt and tools again: llama.cpp
// reuses the start of the prompt that matches the restored one, and a state
// that doesn't match is simply not reused. One per model, saved again when
// the system prompt or tools change.

const STATE_DIR = "wllama-state"

/** What a saved state is for: the model's file, the build that saved it, and the prefix it starts with. */
type StateInfo = {
  url: string
  build: string
  prefix: string
  savedAt: number
}

const debug = (...args: unknown[]) => {
  if (options.debug) console.debug("[wllama state]", ...args)
}

/** The model's files in STATE_DIR. */
const stateFiles = async (model: WllamaTextModel) => {
  const root = await navigator.storage.getDirectory()
  const dir = await root.getDirectoryHandle(STATE_DIR, { create: true })
  const name = model.replace(/[^\w.-]/g, "_")
  return { dir, data: `${name}.bin`, info: `${name}.json` }
}

/** The system prompt and tools of a request, as a short hash and its length. */
function prefixKey(request: ChatRequest) {
  const text = JSON.stringify([
    request.messages.filter((message) => message.role === "system"),
    request.tools ?? []
  ])
  // FNV-1a: a collision only skips a save.
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193)
  }
  return `${(hash >>> 0).toString(16)}:${text.length}`
}

/** The loaded model's build and saved prefix, set by startLoad. */
let state: { model: WllamaTextModel; build: string; prefix?: string } | null =
  null
let saving = false

/** Saves the model's state after a chat, unless one with the same system prompt and tools is saved. */
async function saveState(
  model: WllamaTextModel,
  wllama: Wllama,
  request: ChatRequest
) {
  const current = state
  const prefix = prefixKey(request)
  if (current?.model !== model || current.prefix === prefix || saving) return
  saving = true
  try {
    const start = performance.now()
    const data = await wllama.saveSlotState()
    const saved = performance.now()
    const { dir, data: dataName, info: infoName } = await stateFiles(model)
    const info: StateInfo = {
      url: modelUrl(model),
      build: current.build,
      prefix,
      savedAt: Date.now()
    }
    const write = async (name: string, content: FileSystemWriteChunkType) => {
      const file = await dir.getFileHandle(name, { create: true })
      const writable = await file.createWritable()
      await writable.write(content)
      await writable.close()
    }
    // wllama's copy of the bytes, not its shared memory.
    await write(dataName, data as Uint8Array<ArrayBuffer>)
    await write(infoName, JSON.stringify(info))
    current.prefix = prefix
    debug(
      `saved ${(data.byteLength / 1e6).toFixed(1)} MB in ${Math.round(saved - start)} ms (+${Math.round(performance.now() - saved)} ms to write)`
    )
  } catch (error) {
    debug("not saved:", error)
  } finally {
    saving = false
  }
}

/** Restores the model's saved state, if it's from this build. */
async function restoreState(model: WllamaTextModel, wllama: Wllama) {
  const current = state
  try {
    const start = performance.now()
    const { dir, data, info: infoName } = await stateFiles(model)
    const file = async (name: string) =>
      (await dir.getFileHandle(name)).getFile()
    const info = JSON.parse(await (await file(infoName)).text()) as StateInfo
    if (info.url !== modelUrl(model) || info.build !== current?.build) return
    const bytes = new Uint8Array(await (await file(data)).arrayBuffer())
    const read = performance.now()
    const tokens = await wllama.restoreSlotState(bytes)
    current.prefix = info.prefix
    debug(
      `restored ${tokens} tokens, ${(bytes.byteLength / 1e6).toFixed(1)} MB, in ${Math.round(performance.now() - read)} ms (+${Math.round(read - start)} ms to read)`
    )
  } catch (error) {
    debug("not restored:", error)
  }
}

async function startLoad(
  model: WllamaTextModel,
  wllama: Wllama,
  signal: AbortSignal
) {
  const status = modelStatus(model)
  const url = modelUrl(model)
  const meta = modelMeta(model)
  const compat = needsCompat() ? await compatBuild() : null
  if (compat) useCompat(wllama, compat)
  state = { model, build: `${BUILD_ID} ${compat?.wasm ?? mainWasmUrl()}` }
  // A model already in storage loads from it, without the network.
  const saved = await wllama.modelManager.getModels()
  const cached = saved.some((m) => m.url === url)
  status.set({ state: "loading", progress: 0, cached })
  await wllama.loadModelFromUrl(url, {
    // wllama's default is 1,024 tokens, less than a system prompt with tools.
    n_ctx: meta.context,
    // wllama's default otherwise: WebGPU where the browser has it.
    ...(options.gpu === false && { n_gpu_layers: 0 }),
    // The whole context in sliding-window layers' cache (LFM3's), not just
    // the window, so a follow-up can reuse what was read before: otherwise
    // LFM3 rereads the whole prompt after a tool call. +65 MB for LFM3;
    // nothing for models without such layers.
    swa_full: true,
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
  await restoreState(model, wllama)
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
    { default: mainWasmUrl() },
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
  const chunks = await wllama.createChatCompletion({
    ...request,
    stream: true,
    ...(signal && { abortSignal: signal })
  })
  return (async function* () {
    yield* chunks
    void saveState(model, wllama, request)
  })()
}

/** A whole chat completion from `model`, loading it first. */
export async function completeChat(
  model: WllamaTextModel,
  request: ChatRequest,
  signal?: AbortSignal
): Promise<ChatCompletionResponse> {
  const wllama = await loadModel(model)
  try {
    const response = await wllama.createChatCompletion({
      ...request,
      stream: false,
      ...(signal && { abortSignal: signal })
    })
    void saveState(model, wllama, request)
    return response
  } catch (error) {
    throw contextError(error, modelMeta(model).context)
  }
}
