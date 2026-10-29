/**
 * Runs a local model with Transformers.js on WebGPU, off the main thread so
 * the page stays smooth while it loads and generates. One model at a time:
 * loading another frees the previous one. Transformers.js comes from the CDN
 * at runtime so its native Node dependencies never enter the app build.
 */
import type {
  ChatTurn,
  RunTimings,
  Sampling,
  ToolSpec,
  WorkerRequest,
  WorkerResponse
} from "./local-model-protocol"

const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2"

/* Minimal shapes of the Transformers.js APIs used here. */
type Tensor = {
  dims: Array<number>
  location?: string
  slice: (...ranges: Array<null | [number, number]>) => Tensor
  dispose: () => void
}
/** The model's memory of the text so far: attention keys/values and conv states. */
type Cache = {
  update: (entries: Record<string, Tensor>) => void
  dispose: () => Promise<void>
}
type Tokenizer = {
  apply_chat_template: (
    messages: Array<ChatTurn>,
    options: Record<string, unknown>
  ) => string
  (text: string, options: Record<string, unknown>): { input_ids: Tensor }
}
type Model = {
  generate: (options: Record<string, unknown>) => Promise<unknown>
  forward: (inputs: Record<string, unknown>) => Promise<Record<string, Tensor>>
  dispose: () => Promise<unknown>
}
type StoppingCriteria = { interrupt: () => void; interrupted: boolean }
type TransformersModule = {
  AutoTokenizer: {
    from_pretrained: (
      id: string,
      options?: Record<string, unknown>
    ) => Promise<Tokenizer>
  }
  AutoModelForCausalLM: {
    from_pretrained: (
      id: string,
      options: Record<string, unknown>
    ) => Promise<Model>
  }
  TextStreamer: new (
    tokenizer: Tokenizer,
    options: Record<string, unknown>
  ) => unknown
  InterruptableStoppingCriteria: new () => StoppingCriteria
  DynamicCache: new (entries: Record<string, Tensor>) => Cache
  Tensor: new (type: string, data: unknown, dims: Array<number>) => Tensor
  ones: (dims: Array<number>) => Tensor
  env: {
    fetch: typeof fetch
    backends: {
      onnx: {
        versions: { web: string }
        wasm: { wasmPaths?: string | { mjs: string; wasm: string } }
      }
    }
  }
}

/*
 * ONNX Runtime ships its WebGPU build twice: with Asyncify (the default) and
 * with JSPI. Asyncify rewrites the WebAssembly so it can pause for the GPU,
 * which leaves functions of over 1 MB; Safari's optimizing compiler spent
 * seconds and hundreds of MB on each of them while a reply was generated
 * (on a Mac, the tab peaked at 2,273 MB during a follow-up after a web
 * search, against 1,543 MB with JSPI, and generated ~15% slower). The JSPI
 * build is 16.8 MB instead of 26.9 MB and runs the same code, so it is used
 * wherever the browser has JSPI (Safari 27 and recent Chrome).
 */
function pickJspiBuild(tf: TransformersModule) {
  if (!("Suspending" in WebAssembly)) return
  const onnx = tf.env.backends.onnx
  const dist = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${onnx.versions.web}/dist/`
  onnx.wasm.wasmPaths = {
    mjs: `${dist}ort-wasm-simd-threaded.jspi.mjs`,
    wasm: `${dist}ort-wasm-simd-threaded.jspi.wasm`
  }
}

/*
 * Hugging Face answers requests referred from *.workers.dev with a 404 and no
 * CORS headers, which the browser reports as "Failed to fetch". The page's
 * no-referrer meta tag doesn't reach workers, so every request from here
 * leaves the referrer out itself.
 */
const fetchWithoutReferrer: typeof fetch = (input, init) =>
  fetch(input, { ...init, referrerPolicy: "no-referrer" })

type Runtime = { tf: TransformersModule; tokenizer: Tokenizer; model: Model }

const ctx = globalThis as unknown as {
  postMessage: (message: WorkerResponse) => void
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
}

/*
 * GPU diagnostics, added to generation errors. Safari has no API for memory
 * use, so this counts the WebGPU buffers created and destroyed (nearly all of
 * the model's memory), and records why the device was lost and any GPU error
 * it reported, if those happen.
 */
/** Every live buffer and its size (dev only), for the memory log's breakdown. */
const liveBuffers = new Map<GPUBuffer, number>()

/** Live buffer memory by size, largest totals first: "12x8.4MB=101MB". */
function liveBreakdown() {
  const totals = new Map<number, { count: number; bytes: number }>()
  for (const size of liveBuffers.values()) {
    const entry = totals.get(size) ?? { count: 0, bytes: 0 }
    entry.count++
    entry.bytes += size
    totals.set(size, entry)
  }
  return [...totals]
    .sort((a, b) => b[1].bytes - a[1].bytes)
    .slice(0, 8)
    .map(
      ([size, t]) =>
        `${t.count}x${(size / 1e6).toFixed(2)}MB=${Math.round(t.bytes / 1e6)}MB`
    )
    .join(" ")
}

/** Sizes and usages of large buffers created since the last memory log. */
let createdSizes: Array<string> = []

const gpu = {
  liveBytes: 0,
  peakBytes: 0,
  lost: null as string | null,
  error: null as string | null,
  promptTokens: 0,
  /** Running totals of GPU work, for the dev memory log. */
  churn: {
    buffers: 0,
    bufferBytes: 0,
    /** Buffer requests served from the pool. */
    reused: 0,
    bindGroups: 0,
    /** Bind groups the garbage collector has freed. */
    bindGroupsFreed: 0,
    encoders: 0,
    submits: 0,
    maps: 0,
    uploadBytes: 0
  }
}

/*
 * Every generated token, ONNX Runtime allocates two 4.2 MB storage buffers
 * for LFM3's outputs (at a ~3,000-token prompt) and destroys them a step
 * later: its bucket cache keeps only a few spares per size (20 at 4 MiB, 10
 * at 8 MiB) and destroys the rest. iPhone Safari didn't reclaim the
 * destroyed memory fast enough: 8.4 MB per token, and the page passed its
 * 2 GB limit ~350 tokens into a reply. Destroyed buffers of 1 MB or more are
 * kept and handed back for the next request of the same size and usage
 * instead. (microsoft/onnxruntime#33751)
 */
const POOL_MIN_BYTES = 1 << 20
/**
 * Spares of a size are destroyed once that many pooled requests have gone by
 * without one for it: past 4,096 tokens the 4 MB buffers give way to 8 MB
 * ones, and their spares would otherwise sit in memory for good.
 */
const POOL_STALE_REQUESTS = 512

/** GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE. */
const MAP_READ_OR_WRITE = 0x1 | 0x2
/** Free buffers kept per size and usage. */
const POOL_PER_SIZE = 12

/** The pool a buffer request can be served from, or null. */
function poolKey(descriptor: GPUBufferDescriptor) {
  const mappable = descriptor.usage & MAP_READ_OR_WRITE
  if (
    descriptor.size < POOL_MIN_BYTES ||
    mappable ||
    descriptor.mappedAtCreation
  ) {
    return null
  }
  return `${descriptor.size}/${descriptor.usage}`
}

function trackGpu() {
  const scope = globalThis as unknown as {
    GPUDevice?: { prototype: GPUDevice }
    GPUBuffer?: { prototype: GPUBuffer }
    GPUAdapter?: { prototype: GPUAdapter }
  }
  const device = scope.GPUDevice?.prototype
  const buffer = scope.GPUBuffer?.prototype
  const adapter = scope.GPUAdapter?.prototype
  if (!device || !buffer || !adapter) return
  const sizes = new WeakMap<GPUBuffer, number>()
  const owners = new WeakMap<GPUBuffer, { device: GPUDevice; key: string }>()
  const pools = new WeakMap<GPUDevice, Map<string, Array<GPUBuffer>>>()
  const poolFor = (owner: GPUDevice) => {
    let pool = pools.get(owner)
    if (!pool) pools.set(owner, (pool = new Map()))
    return pool
  }
  // Pooled requests so far, and the last one for each size.
  let requests = 0
  const lastRequested = new Map<string, number>()
  const createBuffer = device.createBuffer
  device.createBuffer = function (descriptor) {
    const key = poolKey(descriptor)
    if (key) {
      lastRequested.set(key, ++requests)
      if (requests % 64 === 0) dropStaleSpares(poolFor(this))
    }
    const reused = key ? poolFor(this).get(key)?.pop() : undefined
    if (reused) {
      gpu.churn.reused++
      return reused
    }
    const created = createBuffer.call(this, descriptor)
    if (key) owners.set(created, { device: this, key })
    gpu.churn.buffers++
    gpu.churn.bufferBytes += descriptor.size
    if (descriptor.size >= 256 * 1024) {
      createdSizes.push(
        `${(descriptor.size / 1e6).toFixed(1)}MB/u${descriptor.usage}`
      )
    }
    sizes.set(created, descriptor.size)
    if (import.meta.env.DEV) liveBuffers.set(created, descriptor.size)
    gpu.liveBytes += descriptor.size
    gpu.peakBytes = Math.max(gpu.peakBytes, gpu.liveBytes)
    return created
  }
  const destroy = buffer.destroy
  /** Destroys a buffer for real. */
  const release = (target: GPUBuffer) => {
    owners.delete(target)
    const size = sizes.get(target)
    if (size !== undefined) {
      gpu.liveBytes -= size
      sizes.delete(target)
      liveBuffers.delete(target)
    }
    destroy.call(target)
  }
  /** Destroys the spares of sizes nothing has asked for in a while. */
  const dropStaleSpares = (pool: Map<string, Array<GPUBuffer>>) => {
    for (const [key, free] of pool) {
      if (requests - (lastRequested.get(key) ?? 0) < POOL_STALE_REQUESTS)
        continue
      for (const spare of free) release(spare)
      pool.delete(key)
    }
  }
  buffer.destroy = function () {
    // A large buffer is kept for the next request of its size, not destroyed.
    const owner = owners.get(this)
    if (owner) {
      const pool = poolFor(owner.device)
      const free = pool.get(owner.key) ?? []
      if (free.length < POOL_PER_SIZE && !free.includes(this)) {
        free.push(this)
        pool.set(owner.key, free)
        return
      }
    }
    release(this)
  }
  if (import.meta.env.DEV) countGpuWork()
  const requestDevice = adapter.requestDevice
  adapter.requestDevice = async function (descriptor) {
    const requested = await requestDevice.call(this, descriptor)
    void requested.lost.then((info) => {
      gpu.lost = `${info.reason}${info.message ? `: ${info.message}` : ""}`
      for (const notify of lostWaiters) notify()
    })
    requested.addEventListener("uncapturederror", (event) => {
      gpu.error = (event as GPUUncapturedErrorEvent).error.message
    })
    return requested
  }
}
trackGpu()

/*
 * Dev-only memory logging. iPhone Safari kills the page at 2 GB, and it has
 * no memory API, so this counts what it can: ONNX Runtime's WebAssembly
 * memory (the WebAssembly memories created here) and the WebGPU buffers.
 */
const wasmMemories = new Set<WebAssembly.Memory>()

function trackWasm() {
  const wasm = WebAssembly as unknown as Record<string, unknown>
  const Memory = WebAssembly.Memory
  wasm.Memory = new Proxy(Memory, {
    construct(target, args: [WebAssembly.MemoryDescriptor]) {
      const memory = Reflect.construct(target, args) as WebAssembly.Memory
      wasmMemories.add(memory)
      return memory
    }
  })
  for (const name of ["instantiate", "instantiateStreaming"]) {
    const original = wasm[name] as
      ((...args: Array<unknown>) => Promise<unknown>) | undefined
    if (!original) continue
    wasm[name] = async function (this: unknown, ...args: Array<unknown>) {
      const result = (await original.apply(this, args)) as {
        instance?: WebAssembly.Instance
        exports?: WebAssembly.Exports
      }
      const exports = result.instance?.exports ?? result.exports ?? {}
      for (const value of Object.values(exports)) {
        if (value instanceof Memory) wasmMemories.add(value)
      }
      return result
    }
  }
}
if (import.meta.env.DEV) trackWasm()

/** Counts WebGPU objects made and work sent, which buffer sizes don't show. */
function countGpuWork() {
  const scope = globalThis as unknown as {
    GPUDevice?: { prototype: GPUDevice }
    GPUQueue?: { prototype: GPUQueue }
    GPUBuffer?: { prototype: GPUBuffer }
  }
  const device = scope.GPUDevice?.prototype
  const queue = scope.GPUQueue?.prototype
  const buffer = scope.GPUBuffer?.prototype
  if (!device || !queue || !buffer) return
  const count = <T extends object, K extends keyof T>(
    target: T,
    name: K,
    onCall: (args: Array<unknown>) => void
  ) => {
    const original = target[name] as unknown as (
      ...args: Array<unknown>
    ) => unknown
    ;(target as Record<K, unknown>)[name] = function (
      this: unknown,
      ...args: Array<unknown>
    ) {
      onCall(args)
      return original.apply(this, args)
    }
  }
  // Bind groups have no destroy(): their memory goes when they're collected.
  const collected = new FinalizationRegistry(() => gpu.churn.bindGroupsFreed++)
  const createBindGroup = device.createBindGroup
  device.createBindGroup = function (descriptor) {
    const bindGroup = createBindGroup.call(this, descriptor)
    gpu.churn.bindGroups++
    collected.register(bindGroup, null)
    return bindGroup
  }
  count(device, "createCommandEncoder", () => gpu.churn.encoders++)
  count(queue, "submit", () => gpu.churn.submits++)
  count(buffer, "mapAsync", () => gpu.churn.maps++)
  count(queue, "writeBuffer", (args) => {
    const data = args[2] as ArrayBufferView | ArrayBuffer | undefined
    gpu.churn.uploadBytes += data?.byteLength ?? 0
  })
}

let lastChurn = { ...gpu.churn }

/** Logs memory use to the page's console, in dev only. */
function logMemory(step: string) {
  if (!import.meta.env.DEV) return
  const mb = (bytes: number) => Math.round(bytes / 1e6)
  let wasmBytes = 0
  for (const memory of wasmMemories) wasmBytes += memory.buffer.byteLength
  // What was made and sent since the last line.
  const c = gpu.churn
  const d = (key: keyof typeof c) => c[key] - lastChurn[key]
  const sizes = new Map<string, number>()
  for (const size of createdSizes) sizes.set(size, (sizes.get(size) ?? 0) + 1)
  createdSizes = []
  const largest = [...sizes]
    .sort((a, b) => parseFloat(b[0]) - parseFloat(a[0]))
    .slice(0, 6)
    .map(([size, n]) => `${n}x${size}`)
    .join(" ")
  const since = `+${d("buffers")} buffers (${mb(d("bufferBytes"))} MB), ${d("reused")} reused, +${d("bindGroups")} bind groups, +${d("encoders")} encoders, +${d("submits")} submits, +${d("maps")} reads, ${mb(d("uploadBytes"))} MB uploaded; large: ${largest || "none"}`
  lastChurn = { ...c }
  const alive = c.bindGroups - c.bindGroupsFreed
  ctx.postMessage({
    type: "log",
    text: `[memory] ${step}: wasm ${mb(wasmBytes)} MB, GPU buffers ${mb(gpu.liveBytes)} MB (peak ${mb(gpu.peakBytes)} MB), ${alive} bind groups not yet collected; since last: ${since}`
  })
}

/** Called when the GPU device is lost. */
const lostWaiters = new Set<() => void>()

/**
 * Runs `work`, failing as soon as the GPU device is lost. When Safari's GPU
 * process crashes, ONNX Runtime can wait forever on a GPU that is gone, and
 * the reply would sit at "Thinking..." with no error.
 */
async function unlessGpuLost<T>(work: Promise<T>): Promise<T> {
  let notify = () => undefined as void
  const lost = new Promise<never>((_, reject) => {
    notify = () =>
      reject(
        new Error(
          "The GPU stopped while the model was running (Safari reset it). Send your message again to reload the model."
        )
      )
  })
  lostWaiters.add(notify)
  try {
    return await Promise.race([work, lost])
  } finally {
    lostWaiters.delete(notify)
  }
}

/**
 * WebKit frees a WebGPU bind group's memory only when its JavaScript object
 * is garbage-collected (there is no destroy()), and until late 2026 a bind
 * group also kept destroyed buffers it pointed at. ONNX Runtime makes ~300
 * bind groups per generated token while the worker's own heap barely grows,
 * so collections rarely ran: iPhone Safari's page grew with every token of a
 * long reply until it passed its 2 GB limit. Allocating throwaway memory
 * makes the engine collect.
 */
function nudgeGarbageCollector() {
  const junk = Array.from({ length: 32 }, () => new ArrayBuffer(1 << 20))
  return junk.length
}

/** What the GPU looked like when something failed, for the error message. */
function gpuReport() {
  const mb = (bytes: number) => Math.round(bytes / 1e6)
  return [
    `GPU buffers: ${mb(gpu.liveBytes)} MB in use, ${mb(gpu.peakBytes)} MB peak`,
    `prompt: ${gpu.promptTokens} tokens`,
    gpu.lost && `device lost: ${gpu.lost}`,
    gpu.error && `GPU error: ${gpu.error}`
  ]
    .filter(Boolean)
    .join("; ")
}

/*
 * The weights file (`onnx/model_q4f16.onnx_data`, 0.76-1.09 GB) is cached in
 * the Origin Private File System instead of Transformers.js's Cache Storage:
 * Transformers.js 4.3 holds the whole download in memory and copies it again
 * to cache it, which goes past iPhone Safari's 2 GB page limit. Downloading
 * straight to disk and streaming it back in slice by slice avoids both.
 * Transformers.js still fetches and caches the small files itself.
 */
const WEIGHTS = "model_q4f16.onnx_data"

/**
 * A model id is a Hugging Face repo, optionally pinned to a commit:
 * "owner/repo@revision". Pinning makes browsers fetch that version's files
 * instead of reusing cached ones from an older version.
 */
function parseModelId(modelId: string) {
  const [repo, revision = "main"] = modelId.split("@")
  return { repo, revision }
}

/**
 * The weights file on disk is named by repo and, for a pinned model, by
 * revision, so a new version is downloaded fresh rather than reusing what an
 * older one saved.
 */
const repoFilePrefix = (repo: string) => `${repo.replace(/[^\w.-]+/g, "_")}.`

const weightsFiles = (modelId: string) => {
  const { repo, revision } = parseModelId(modelId)
  const version = revision === "main" ? "" : `${revision.slice(0, 12)}.`
  const file = `${repoFilePrefix(repo)}${version}${WEIGHTS}`
  return {
    url: `https://huggingface.co/${repo}/resolve/${revision}/onnx/${WEIGHTS}`,
    file,
    /** Written once the download is complete. */
    done: `${file}.done`
  }
}

/** OPFS sync access handles (worker-only, not in the DOM typings). */
type SyncAccessHandle = {
  read: (buffer: Uint8Array, options: { at: number }) => number
  write: (buffer: Uint8Array, options: { at: number }) => number
  truncate: (size: number) => void
  getSize: () => number
  flush: () => void
  close: () => void
}
const openSync = (file: FileSystemFileHandle) =>
  (
    file as unknown as {
      createSyncAccessHandle: () => Promise<SyncAccessHandle>
    }
  ).createSyncAccessHandle()

async function hasFile(dir: FileSystemDirectoryHandle, name: string) {
  try {
    await dir.getFileHandle(name)
    return true
  } catch {
    return false
  }
}

/** Whether the weights are already downloaded to disk. */
async function isDownloaded(modelId: string) {
  try {
    const dir = await navigator.storage.getDirectory()
    return await hasFile(dir, weightsFiles(modelId).done)
  } catch {
    return false
  }
}

/** Downloads the weights to disk, writing each network chunk as it arrives. */
async function downloadWeights(
  modelId: string,
  onProgress: (progress: number) => void
) {
  const files = weightsFiles(modelId)
  const dir = await navigator.storage.getDirectory()
  if (await hasFile(dir, files.done)) return
  const res = await fetchWithoutReferrer(files.url)
  if (!res.ok || !res.body) {
    throw new Error(`Couldn't download the model (HTTP ${res.status}).`)
  }
  const total = Number(res.headers.get("content-length")) || 0
  const handle = await openSync(
    await dir.getFileHandle(files.file, { create: true })
  )
  try {
    handle.truncate(0)
    let written = 0
    const reader = res.body.getReader()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      written += handle.write(value, { at: written })
      if (total) onProgress(written / total)
    }
    handle.flush()
  } finally {
    handle.close()
  }
  await dir.getFileHandle(files.done, { create: true })
}

/**
 * Opens the weights on disk as a stand-in for the whole buffer. ONNX Runtime
 * only asks it for its size and for one tensor's slice at a time, so each
 * slice is read from disk when asked and the file never sits in memory.
 * Call close() once the model is created.
 */
async function openWeights(modelId: string) {
  const dir = await navigator.storage.getDirectory()
  const handle = await openSync(
    await dir.getFileHandle(weightsFiles(modelId).file)
  )
  const size = handle.getSize()
  // Every slice is read into the same buffer, grown to the largest tensor:
  // ONNX Runtime copies a slice to the GPU (or its own memory) as soon as it
  // gets it. New buffers per slice left the whole file's worth of them for
  // the garbage collector, which iPhone Safari seemed to keep around,
  // compressed, counting against the tab's memory limit.
  let scratch = new ArrayBuffer(0)
  const view = new Uint8Array(0)
  Object.defineProperties(view, {
    byteLength: { get: () => size },
    length: { get: () => size },
    subarray: {
      value: (begin = 0, end = size) => {
        const length = end - begin
        if (scratch.byteLength < length) scratch = new ArrayBuffer(length)
        const slice = new Uint8Array(scratch, 0, length)
        handle.read(slice, { at: begin })
        return slice
      }
    }
  })
  return {
    data: view,
    close: () => {
      handle.close()
      scratch = new ArrayBuffer(0)
    }
  }
}

/**
 * Removes weights earlier versions of the app stored elsewhere, and saved
 * weights of other versions of this model.
 */
async function removeOldCopies(modelId: string) {
  try {
    const cache = await caches.open("transformers-cache")
    await cache.delete(weightsFiles(modelId).url)
    const { repo } = parseModelId(modelId)
    const current = weightsFiles(modelId)
    const dir = await navigator.storage.getDirectory()
    for await (const name of (
      dir as unknown as { keys: () => AsyncIterable<string> }
    ).keys()) {
      const otherVersion =
        name.startsWith(repoFilePrefix(repo)) &&
        name !== current.file &&
        name !== current.done
      if (name.startsWith("lfm3-1.7b-") || otherVersion) {
        await dir.removeEntry(name)
      }
    }
  } catch {
    // Nothing to clean up.
  }
}

let current: { modelId: string; runtime: Promise<Runtime> } | null = null

async function loadRuntime(modelId: string): Promise<Runtime> {
  if (!("gpu" in navigator)) {
    throw new Error(
      "This model needs WebGPU, which this browser doesn't support."
    )
  }
  const cached = await isDownloaded(modelId)
  ctx.postMessage({ type: "progress", modelId, progress: 0, cached })
  // Whole percents only; the download stream reports every chunk.
  let lastPercent = -1
  await downloadWeights(modelId, (progress) => {
    const percent = Math.floor(progress * 100)
    if (percent === lastPercent) return
    lastPercent = percent
    ctx.postMessage({
      type: "progress",
      modelId,
      progress: percent / 100,
      cached: false
    })
  })
  ctx.postMessage({ type: "progress", modelId, progress: 0, cached: true })
  const tf = (await import(
    /* @vite-ignore */ TRANSFORMERS_URL
  )) as TransformersModule
  tf.env.fetch = fetchWithoutReferrer
  pickJspiBuild(tf)
  const { repo, revision } = parseModelId(modelId)
  const tokenizer = await tf.AutoTokenizer.from_pretrained(repo, { revision })
  const weights = await openWeights(modelId)
  try {
    const model = await tf.AutoModelForCausalLM.from_pretrained(repo, {
      revision,
      dtype: "q4f16",
      device: "webgpu",
      // Weights come from disk (above), not Transformers.js's download.
      use_external_data_format: 0,
      session_options: {
        externalData: [{ path: WEIGHTS, data: weights.data }],
        // onnxruntime's default buffer cache ("bucket") reuses GPU buffers
        // between steps. "lazyRelease" frees them after every step instead,
        // which creates and destroys ~27 MB of buffers per token for LFM3;
        // WebKit releases destroyed buffers late, so a long reply climbed
        // past an iPhone's memory limit (on a Mac: 1,672 MB at 1,000 tokens
        // and climbing, against 1,137 MB and flat with bucket).
        executionProviders: ["webgpu"]
      }
    })
    return { tf, tokenizer, model }
  } finally {
    weights.close()
  }
}

/** Loads `modelId`, freeing whichever model was loaded before. */
function load(modelId: string) {
  if (current?.modelId === modelId) return current.runtime
  const previous = current?.runtime
  const runtime = (async () => {
    await previous?.then((r) => r.model.dispose()).catch(() => undefined)
    void removeOldCopies(modelId)
    return loadRuntime(modelId)
  })()
  current = { modelId, runtime }
  runtime.then(
    () => {
      logMemory(`loaded ${modelId}`)
      ctx.postMessage({ type: "ready", modelId })
    },
    (error: unknown) => {
      if (current?.runtime === runtime) current = null
      ctx.postMessage({
        type: "load-error",
        modelId,
        message: errorMessage(error)
      })
    }
  )
  return runtime
}

const running = new Map<string, StoppingCriteria>()

/** End-of-turn tokens, dropped when special tokens are kept. */
const END_TOKENS = /<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>/g

/**
 * The most tokens a chat can hold, prompt and reply together: the same as
 * Apple's on-device model. Past 4,096 tokens ONNX Runtime's buffer cache
 * moves LFM3's attention memory to 8 MiB buffers, which hold exactly 8,192;
 * one more token would move it to 12 MiB, past what iPhone Safari allows.
 */
const CONTEXT_LIMIT = 8192
/** Below this much room for a reply, the chat is too long to answer. */
const MIN_REPLY_TOKENS = 32

/** A chat past the context limit: shown as is, without GPU diagnostics. */
class ContextLimitError extends Error {}

/**
 * Prompt tokens read per step. iPhone Safari's GPU process crashed reading
 * an 868-token prompt in one step (a web search's results, or a pasted
 * article), while short prompts and long replies were fine.
 */
const PROMPT_CHUNK = 256

/** Transformers.js's names for a model output's place in the cache. */
const cacheName = (output: string) =>
  output
    .replace("present_conv", "past_conv")
    .replace("present_ssm", "past_ssm")
    .replace("present_recurrent", "past_recurrent")
    .replace("present", "past_key_values")

/**
 * Reads all but the end of a long prompt into the model's cache, a chunk at
 * a time, so no single GPU step grows with the prompt. generate() reads the
 * rest and replies. Null for a prompt short enough to read in one step.
 */
async function readPrompt(
  tf: TransformersModule,
  model: Model,
  inputIds: Tensor,
  stopping: StoppingCriteria
) {
  const length = inputIds.dims[1]
  // At least one token is left for generate(), which skips what the cache
  // already holds; with none left it would read the whole prompt again.
  const end = Math.floor((length - 1) / PROMPT_CHUNK) * PROMPT_CHUNK
  if (end === 0) return null
  let cache: Cache | null = null
  try {
    for (let start = 0; start < end; start += PROMPT_CHUNK) {
      if (stopping.interrupted) break
      const stop = start + PROMPT_CHUNK
      const outputs = await model.forward({
        input_ids: inputIds.slice(null, [start, stop]),
        attention_mask: tf.ones([1, stop]),
        ...(cache && { past_key_values: cache }),
        // Logits for the last token only (the default here is every token).
        num_logits_to_keep: new tf.Tensor("int64", [1n], [])
      })
      const entries: Record<string, Tensor> = {}
      for (const [name, tensor] of Object.entries(outputs)) {
        if (name.startsWith("present")) entries[cacheName(name)] = tensor
        else if (tensor.location === "gpu-buffer") tensor.dispose()
      }
      if (cache) cache.update(entries)
      else cache = new tf.DynamicCache(entries)
      nudgeGarbageCollector()
      logMemory(`prompt read ${stop} of ${length} tokens`)
    }
    return cache
  } catch (error) {
    await cache?.dispose()
    throw error
  }
}

/** How often a reply's counts go to the page while it generates. */
const STATS_INTERVAL_MS = 250

async function generate(
  id: string,
  chat: Array<ChatTurn>,
  tools: Array<ToolSpec>,
  thinking: boolean,
  sampling: Sampling
): Promise<RunTimings> {
  // Time to first token counts from here: it includes reading the prompt.
  const requested = performance.now()
  if (!current) throw new Error("No model loaded.")
  const loaded = current
  const { tf, tokenizer, model } = await loaded.runtime
  // Models with a thinking mode reason in a <think> block before answering.
  const prompt = tokenizer.apply_chat_template(chat, {
    ...(tools.length > 0 && { tools }),
    add_generation_prompt: true,
    tokenize: false,
    enable_thinking: thinking
  })
  // Some templates open the <think> block in the prompt itself; start the
  // reply with it too so the thread shows what follows as thinking.
  if (prompt.trimEnd().endsWith("<think>")) {
    ctx.postMessage({ type: "token", id, text: "<think>" })
  }
  const inputs = tokenizer(prompt, { add_special_tokens: false })
  gpu.promptTokens = inputs.input_ids.dims[1]
  // The reply gets what's left of the context limit.
  const room = CONTEXT_LIMIT - gpu.promptTokens
  if (room < MIN_REPLY_TOKENS) {
    throw new ContextLimitError(
      `This chat is too long for the model: ${gpu.promptTokens.toLocaleString("en-US")} tokens, and it can read ${CONTEXT_LIMIT.toLocaleString("en-US")} at most, reply included. Start a new chat to keep going.`
    )
  }
  // Thinking takes many more tokens before the answer starts.
  const wanted = thinking ? 4096 : 1024
  const maxNewTokens = Math.min(wanted, room)
  const stopping = new tf.InterruptableStoppingCriteria()
  running.set(id, stopping)
  // With tools, keep special tokens: LFM2.5 marks its tool calls with them.
  const keepSpecial = tools.length > 0
  // Every generated token, counted as the streamer gets it (text arrives
  // word by word, later than its tokens).
  let outputTokens = 0
  let firstToken: number | null = null
  let lastToken = 0
  const timings = (): RunTimings => ({
    inputTokens: inputs.input_ids.dims[1],
    // Every run reads the whole prompt, so none of it is cached. Reusing the
    // cache between runs would report its length here as cachedTokens.
    outputTokens,
    firstTokenMs: firstToken === null ? null : firstToken - requested,
    decodeMs: firstToken === null ? 0 : lastToken - firstToken
  })
  // Counts so far for the stats line: the prompt's now, then the reply's at
  // its first token and every STATS_INTERVAL_MS after.
  let reported = 0
  const report = () => {
    reported = performance.now()
    ctx.postMessage({ type: "stats", id, stats: timings() })
  }
  report()
  const streamer = new tf.TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: !keepSpecial,
    token_callback_function: (tokens: Array<bigint>) => {
      lastToken = performance.now()
      firstToken ??= lastToken
      outputTokens += tokens.length
      if (
        firstToken === lastToken ||
        lastToken - reported >= STATS_INTERVAL_MS
      ) {
        report()
      }
      if (outputTokens % 16 === 0) nudgeGarbageCollector()
      if (outputTokens % 32 === 0) {
        logMemory(`reply ${outputTokens} tokens (prompt ${gpu.promptTokens})`)
      }
    },
    callback_function: (raw: string) => {
      const text = keepSpecial ? raw.replace(END_TOKENS, "") : raw
      if (text) ctx.postMessage({ type: "token", id, text })
    }
  })
  let cache: Cache | null = null
  try {
    cache = await unlessGpuLost(
      readPrompt(tf, model, inputs.input_ids, stopping)
    )
    if (stopping.interrupted) return timings()
    await unlessGpuLost(
      model.generate({
        ...inputs,
        ...(cache && {
          past_key_values: cache,
          attention_mask: tf.ones(inputs.input_ids.dims)
        }),
        max_new_tokens: maxNewTokens,
        do_sample: true,
        ...sampling,
        streamer,
        stopping_criteria: [stopping]
      })
    )
  } catch (error) {
    // The model's GPU memory is gone with the device: load it again next time.
    if (gpu.lost && current === loaded) current = null
    throw error
  } finally {
    running.delete(id)
    // generate() keeps a cache it was given; this one is done.
    await cache?.dispose().catch(() => undefined)
    logMemory(`reply finished, ${outputTokens} tokens`)
    if (import.meta.env.DEV) {
      ctx.postMessage({
        type: "log",
        text: `[memory] live buffers: ${liveBreakdown()}`
      })
    }
  }
  // Cut off by the context limit, not by the reply's own length limit.
  if (
    maxNewTokens < wanted &&
    outputTokens >= maxNewTokens &&
    !stopping.interrupted
  ) {
    report()
    throw new ContextLimitError(
      `The reply stopped at the model's limit of ${CONTEXT_LIMIT.toLocaleString("en-US")} tokens for a chat. Start a new chat to keep going.`
    )
  }
  return timings()
}

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

ctx.onmessage = ({ data }) => {
  if (data.type === "load") {
    load(data.modelId).catch(() => undefined)
  } else if (data.type === "generate") {
    generate(data.id, data.chat, data.tools, data.thinking, data.sampling)
      .then((stats) => ctx.postMessage({ type: "done", id: data.id, stats }))
      .catch((error: unknown) =>
        ctx.postMessage({
          type: "error",
          id: data.id,
          message:
            error instanceof ContextLimitError
              ? error.message
              : `${errorMessage(error)} (${gpuReport()})`
        })
      )
  } else {
    running.get(data.id)?.interrupt()
  }
}
