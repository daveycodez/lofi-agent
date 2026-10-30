/**
 * The page's side of wllama: llama.cpp compiled to WebAssembly, which runs
 * in wllama's own Web Worker (several threads where the page is
 * cross-origin isolated). One model at a time: loading another frees the
 * previous one. Models download once to the Origin Private File System,
 * and load from there afterwards, offline included.
 */
import { LoggerWithoutDebug, Wllama } from "@wllama/wllama/esm/index.js"
import type { RawCompletionChunk } from "@wllama/wllama/esm/index.js"

import { modelMeta, modelUrl } from "./model-meta.js"
import type { Sampling, WllamaTextModel } from "./model-meta.js"
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
  /**
   * CPU threads: three quarters of the cores the browser reports by
   * default, where the page is cross-origin isolated; one otherwise
   * (WebAssembly threads need SharedArrayBuffer).
   */
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
  return !("Suspending" in WebAssembly) || !WebAssembly.validate(MEMORY64_MODULE)
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
      throw new Error(`Couldn't load wllama's compat build (${response.status}).`)
    }
    return `${await response.text()}\n`
  })
  compatCode.catch(() => {
    compatCode = undefined
  })
  return { wasm, worker: { code: await compatCode } }
}

/**
 * Threads to run on (see `RuntimeOptions.threads`). Not every core:
 * llama.cpp's threads wait for each other at every step, so one slowed by
 * the page's own work slows them all. In lofi-agent on an M3 Ultra (32
 * cores), LFM2.5 1.2B wrote 76 tokens/s on 16 threads, 76 on 24, 55 on 28
 * and 33 on 32, and read prompts at 154, 221, 223 and 217 tokens/s.
 */
export function cpuThreads() {
  if (options.threads) return options.threads
  const isolated =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated
  if (!isolated) return 1
  return Math.max(1, Math.floor(((navigator.hardwareConcurrency || 4) * 3) / 4))
}

/**
 * Prompts are read in steps of this many tokens. llama.cpp saves the
 * model's state ahead of the last step of each prompt (LFM2's convolution
 * state can't be rolled back otherwise), so a smaller step means a new
 * chat, or the next turn after a template dropped earlier thinking, reuses
 * more of the prompt: within 128 tokens of where it stops matching. Reading
 * is as fast as with llama.cpp's 512.
 */
const PROMPT_STEP = 128

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
    n_ctx: meta.context,
    n_threads: cpuThreads(),
    n_ubatch: PROMPT_STEP,
    // On the CPU, with llama.cpp's WebAssembly kernels, the same in every
    // browser; wllama would otherwise put the model on WebGPU where there is.
    n_gpu_layers: 0,
    // One chat at a time, holding the whole context. A new chat reuses the
    // last one's state saved within its system prompt and tools (see
    // PROMPT_STEP).
    n_parallel: 1,
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

/** Token counts and timings of one generation. */
export type RunTimings = {
  /** Tokens of the prompt, including any already in the model's cache. */
  inputTokens: number
  /** Tokens at the start of the prompt the model had read on an earlier run. */
  cachedTokens: number
  /** Tokens the model generated. */
  outputTokens: number
  /** From the request to the first generated text, in ms; null if none. */
  firstTokenMs: number | null
  /** Reading the part of the prompt that wasn't cached, in ms. */
  promptMs: number | null
  /** Generating, in ms, and the tokens generated meanwhile. */
  decodeMs: number
  decodeTokens: number
  /** Whether the reply stopped at its token limit. */
  truncated: boolean
}

const count = new Intl.NumberFormat("en-US")

/** llama.cpp's error for a prompt past the context limit, said plainly. */
function contextError(error: unknown, context: number) {
  const tokens = /request \((\d+) tokens\) exceeds the available context/.exec(
    errorText(error)
  )?.[1]
  if (!tokens) return error
  return new Error(
    `This session is too long for the model: ${count.format(Number(tokens))} tokens, and it can read ${count.format(context)} at most, reply included. Start a new session to keep going.`
  )
}

/** Special tokens that stay in the reply's text (llama.cpp drops them otherwise). */
const KEPT_TOKENS = [
  "<|tool_call_start|>",
  "<|tool_call_end|>",
  "<think>",
  "</think>"
]

/**
 * Runs one generation with the loaded model, yielding its text as it comes
 * (thinking and tool-call markup included, as the model writes them).
 * llama.cpp reuses the start of the prompt it read on the previous run.
 */
export async function* generate(request: {
  model: WllamaTextModel
  prompt: string
  sampling: Sampling
  maxTokens: number
  /** Texts to stop at; they aren't part of the text yielded. */
  stop?: Array<string>
  signal?: AbortSignal
}): AsyncGenerator<string, RunTimings> {
  const wllama = await loadModel(request.model)
  const { context } = modelMeta(request.model)
  const requested = performance.now()
  let firstToken: number | null = null
  let last: RawCompletionChunk | undefined
  const { temperature, top_k, top_p, repeat_penalty } = request.sampling
  const stream = await wllama.createCompletion({
    prompt: request.prompt,
    stream: true,
    max_tokens: request.maxTokens,
    temperature,
    top_k,
    ...(top_p !== undefined && { top_p }),
    ...(request.stop && { stop: request.stop }),
    ...(request.signal && { abortSignal: request.signal }),
    // llama-server's own request fields, which wllama passes through.
    ...({
      repeat_penalty,
      cache_prompt: true,
      preserved_tokens: KEPT_TOKENS
    } as object)
  })
  try {
    for await (const chunk of stream) {
      last = chunk
      const piece = chunk.choices?.[0]?.text
      if (!piece) continue
      firstToken ??= performance.now()
      yield piece
    }
  } catch (error) {
    // Stopped: the reply so far stands.
    if (!request.signal?.aborted) throw contextError(error, context)
  }
  const usage = last?.usage
  const timings = last?.timings
  const inputTokens = usage?.prompt_tokens ?? timings?.prompt_n ?? 0
  const outputTokens = usage?.completion_tokens ?? timings?.predicted_n ?? 0
  // Cut off by the context limit, not by the reply's own length limit.
  if (
    last?.choices?.[0]?.finish_reason === "length" &&
    inputTokens + outputTokens >= context
  ) {
    throw new Error(
      `The reply stopped at the model's limit of ${count.format(context)} tokens for a session. Start a new session to keep going.`
    )
  }
  return {
    inputTokens,
    cachedTokens: timings?.cache_n ?? 0,
    outputTokens,
    firstTokenMs: firstToken === null ? null : firstToken - requested,
    promptMs: timings ? timings.prompt_ms : null,
    decodeMs: timings?.predicted_ms ?? 0,
    decodeTokens: timings?.predicted_n ?? 0,
    truncated: last?.choices?.[0]?.finish_reason === "length"
  }
}
