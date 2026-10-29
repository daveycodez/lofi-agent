/**
 * Runs a local model with Transformers.js on WebGPU, off the main thread so
 * the page stays smooth while it loads and generates. One model at a time:
 * loading another frees the previous one. Transformers.js comes from the CDN
 * at runtime so its native Node dependencies never enter the app build.
 */
import type {
  ChatTurn,
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
  slice: (...ranges: Array<null | [number, number]>) => Tensor
}
type Tokenizer = {
  apply_chat_template: (
    messages: Array<ChatTurn>,
    options: Record<string, unknown>
  ) => string
  (
    text: string,
    options: Record<string, unknown>
  ): { input_ids: Tensor; attention_mask: Tensor }
}
/** The model's cache (attention keys/values and conv states), on the GPU. */
type Cache = { dispose: () => Promise<unknown> }
type Model = {
  generate: (options: Record<string, unknown>) => Promise<unknown>
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
}
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
const gpu = {
  liveBytes: 0,
  peakBytes: 0,
  lost: null as string | null,
  error: null as string | null,
  promptTokens: 0
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
  const createBuffer = device.createBuffer
  device.createBuffer = function (descriptor) {
    const created = createBuffer.call(this, descriptor)
    sizes.set(created, descriptor.size)
    gpu.liveBytes += descriptor.size
    gpu.peakBytes = Math.max(gpu.peakBytes, gpu.liveBytes)
    return created
  }
  const destroy = buffer.destroy
  buffer.destroy = function () {
    const size = sizes.get(this)
    if (size !== undefined) {
      gpu.liveBytes -= size
      sizes.delete(this)
    }
    return destroy.call(this)
  }
  const requestDevice = adapter.requestDevice
  adapter.requestDevice = async function (descriptor) {
    const requested = await requestDevice.call(this, descriptor)
    void requested.lost.then((info) => {
      gpu.lost = `${info.reason}${info.message ? `: ${info.message}` : ""}`
    })
    requested.addEventListener("uncapturederror", (event) => {
      gpu.error = (event as GPUUncapturedErrorEvent).error.message
    })
    return requested
  }
}
trackGpu()

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
  // Hugging Face rejects downloads referred from *.workers.dev.
  const res = await fetch(files.url, { referrerPolicy: "no-referrer" })
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
  const view = new Uint8Array(0)
  Object.defineProperties(view, {
    byteLength: { get: () => size },
    length: { get: () => size },
    subarray: {
      value: (begin = 0, end = size) => {
        const slice = new Uint8Array(end - begin)
        handle.read(slice, { at: begin })
        return slice
      }
    }
  })
  return { data: view, close: () => handle.close() }
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
        // Free GPU buffers after each run instead of keeping every size used:
        // reading a long prompt otherwise leaves hundreds of MB cached, more
        // than an iPhone's memory limit allows.
        executionProviders: [
          { name: "webgpu", storageBufferCacheMode: "lazyRelease" }
        ]
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
    () => ctx.postMessage({ type: "ready", modelId }),
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

/**
 * Prompt tokens read per step when reading a long prompt. iPhones get much
 * smaller steps: their GPU drops the device ("device lost") partway through
 * a 256-token step even though memory is far from full, while ~50-token
 * prompts work, so each step is kept to the size that works.
 */
const PREFILL_CHUNK = /iPhone|iPad|iPod/.test(navigator.userAgent) ? 48 : 256

/**
 * Reads all but the last token of a long prompt into the model's cache,
 * PREFILL_CHUNK tokens at a time. Reading it in one pass needs GPU memory
 * that grows with the square of its length (the attention scores), which
 * runs past an iPhone's limit at a few hundred tokens. generate() then picks
 * up from the cache without reading the prompt again. Returns null for short
 * prompts, which generate() reads in one go as usual.
 */
async function prefill(
  model: Model,
  inputs: { input_ids: Tensor; attention_mask: Tensor },
  stopping: StoppingCriteria
): Promise<Cache | null> {
  const length = inputs.input_ids.dims[1]
  if (length <= PREFILL_CHUNK) return null
  let cache: Cache | null = null
  // The last prompt token is left for generate(), which needs at least one.
  for (let start = 0; start < length - 1; start += PREFILL_CHUNK) {
    if (stopping.interrupted) break
    const end = Math.min(start + PREFILL_CHUNK, length - 1)
    const out = (await model.generate({
      input_ids: inputs.input_ids.slice(null, [0, end]),
      attention_mask: inputs.attention_mask.slice(null, [0, end]),
      past_key_values: cache,
      max_new_tokens: 1,
      do_sample: false,
      return_dict_in_generate: true
    })) as { past_key_values: Cache }
    // The same cache object, updated in place with this chunk.
    cache = out.past_key_values
  }
  return cache
}

/** End-of-turn tokens, dropped when special tokens are kept. */
const END_TOKENS = /<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>/g

async function generate(
  id: string,
  chat: Array<ChatTurn>,
  tools: Array<ToolSpec>,
  thinking: boolean,
  sampling: Sampling
) {
  if (!current) throw new Error("No model loaded.")
  const { tf, tokenizer, model } = await current.runtime
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
  const stopping = new tf.InterruptableStoppingCriteria()
  running.set(id, stopping)
  // With tools, keep special tokens: LFM2.5 marks its tool calls with them.
  const keepSpecial = tools.length > 0
  const streamer = new tf.TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: !keepSpecial,
    callback_function: (raw: string) => {
      const text = keepSpecial ? raw.replace(END_TOKENS, "") : raw
      if (text) ctx.postMessage({ type: "token", id, text })
    }
  })
  let cache: Cache | null = null
  try {
    cache = await prefill(model, inputs, stopping)
    if (stopping.interrupted) return
    await model.generate({
      ...inputs,
      // Continues from the prompt already in the cache, if any.
      ...(cache && { past_key_values: cache }),
      // Thinking takes many more tokens before the answer starts.
      max_new_tokens: thinking ? 4096 : 1024,
      do_sample: true,
      ...sampling,
      streamer,
      stopping_criteria: [stopping]
    })
  } finally {
    running.delete(id)
    // generate() leaves a cache it was given alive; free its GPU memory.
    await cache?.dispose().catch(() => undefined)
  }
}

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

ctx.onmessage = ({ data }) => {
  if (data.type === "load") {
    load(data.modelId).catch(() => undefined)
  } else if (data.type === "generate") {
    generate(data.id, data.chat, data.tools, data.thinking, data.sampling)
      .then(() => ctx.postMessage({ type: "done", id: data.id }))
      .catch((error: unknown) =>
        ctx.postMessage({
          type: "error",
          id: data.id,
          message: `${errorMessage(error)} (${gpuReport()})`
        })
      )
  } else {
    running.get(data.id)?.interrupt()
  }
}
