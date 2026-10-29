/**
 * Runs LFM3-1.7B off the main thread so the page stays smooth while the model
 * loads and generates. Transformers.js comes from the CDN at runtime so its
 * native Node dependencies never enter the app (or Cloudflare) build.
 */
import { MODEL_ID } from "./lfm3-protocol"
import type { ChatTurn, WorkerRequest, WorkerResponse } from "./lfm3-protocol"

const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2"
/* Minimal shapes of the Transformers.js APIs used here. */
type Tensor = { dims: Array<number> }
type Tokenizer = {
  apply_chat_template: (
    messages: Array<ChatTurn>,
    options: Record<string, unknown>
  ) => string
  (text: string, options: Record<string, unknown>): { input_ids: Tensor }
}
type Model = {
  generate: (options: Record<string, unknown>) => Promise<unknown>
}
type StoppingCriteria = { interrupt: () => void }
type TransformersModule = {
  AutoTokenizer: { from_pretrained: (id: string) => Promise<Tokenizer> }
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

const ctx = globalThis as unknown as {
  postMessage: (message: WorkerResponse) => void
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
}

/** The weights file: its name inside the ONNX graph, and where to download it. */
const WEIGHTS_NAME = "model_q4f16.onnx_data"
const WEIGHTS_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/${WEIGHTS_NAME}`
/** Where the weights live on disk, plus a marker written once they're complete. */
const OPFS_FILE = "lfm3-1.7b-q4f16.onnx_data"
const OPFS_DONE = `${OPFS_FILE}.done`
/** Read and write in 64 MB pieces so no single transfer is huge. */
const CHUNK = 64 * 1024 * 1024

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

/**
 * Downloads the 1.09 GB weights into the Origin Private File System, writing
 * each network chunk to disk as it arrives. Safari crashes when a single
 * gigabyte-sized response goes through the Cache API, so Transformers.js's
 * own caching can't be used for this file. Returns false if already on disk.
 */
async function downloadWeights(onProgress: (progress: number) => void) {
  const dir = await navigator.storage.getDirectory()
  if (await hasFile(dir, OPFS_DONE)) return false
  // Hugging Face rejects downloads referred from *.workers.dev.
  const res = await fetch(WEIGHTS_URL, { referrerPolicy: "no-referrer" })
  if (!res.ok || !res.body) {
    throw new Error(`Couldn't download the model (HTTP ${res.status}).`)
  }
  const total = Number(res.headers.get("content-length")) || 0
  const handle = await openSync(
    await dir.getFileHandle(OPFS_FILE, { create: true })
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
  await dir.getFileHandle(OPFS_DONE, { create: true })
  return true
}

/** Reads the weights from disk into one buffer, in pieces. */
async function readWeights() {
  const dir = await navigator.storage.getDirectory()
  const handle = await openSync(await dir.getFileHandle(OPFS_FILE))
  try {
    const buffer = new Uint8Array(handle.getSize())
    for (let at = 0; at < buffer.length; at += CHUNK) {
      handle.read(buffer.subarray(at, Math.min(at + CHUNK, buffer.length)), {
        at
      })
    }
    return buffer
  } finally {
    handle.close()
  }
}

/** Frees the 1 GB copy an earlier version kept in Cache Storage. */
async function dropOldCacheCopy() {
  try {
    const cache = await caches.open("transformers-cache")
    await cache.delete(WEIGHTS_URL)
  } catch {
    // Nothing to clean up.
  }
}

let runtime: Promise<{
  tf: TransformersModule
  tokenizer: Tokenizer
  model: Model
}> | null = null

function load() {
  runtime ??= (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("LFM3 needs WebGPU, which this browser doesn't support.")
    }
    // Download to disk first (whole percents only, the stream reports
    // per chunk), then load from disk.
    let lastPercent = -1
    await downloadWeights((progress) => {
      const percent = Math.floor(progress * 100)
      if (percent === lastPercent) return
      lastPercent = percent
      ctx.postMessage({
        type: "progress",
        progress: percent / 100,
        cached: false
      })
    })
    ctx.postMessage({ type: "progress", progress: 0, cached: true })
    void dropOldCacheCopy()
    const tf = (await import(
      /* @vite-ignore */ TRANSFORMERS_URL
    )) as TransformersModule
    const tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL_ID)
    // Hand ONNX Runtime the weights directly; Transformers.js would
    // otherwise download and cache them itself.
    const model = await tf.AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu",
      use_external_data_format: 0,
      session_options: {
        externalData: [{ path: WEIGHTS_NAME, data: await readWeights() }]
      }
    })
    ctx.postMessage({ type: "ready" })
    return { tf, tokenizer, model }
  })().catch((error: unknown) => {
    runtime = null
    throw error
  })
  return runtime
}

const running = new Map<string, StoppingCriteria>()

async function generate(id: string, chat: Array<ChatTurn>) {
  const { tf, tokenizer, model } = await load()
  // Thinking off: it is slow on small devices and can run out of tokens.
  const prompt = tokenizer.apply_chat_template(chat, {
    add_generation_prompt: true,
    tokenize: false,
    enable_thinking: false
  })
  const inputs = tokenizer(prompt, { add_special_tokens: false })
  const stopping = new tf.InterruptableStoppingCriteria()
  running.set(id, stopping)
  const streamer = new tf.TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => {
      if (text) ctx.postMessage({ type: "token", id, text })
    }
  })
  try {
    await model.generate({
      ...inputs,
      max_new_tokens: 1024,
      do_sample: true,
      temperature: 0.1,
      top_k: 50,
      repetition_penalty: 1.1,
      streamer,
      stopping_criteria: [stopping]
    })
  } finally {
    running.delete(id)
  }
}

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

ctx.onmessage = ({ data }) => {
  if (data.type === "load") {
    load().catch((error: unknown) =>
      ctx.postMessage({ type: "load-error", message: errorMessage(error) })
    )
  } else if (data.type === "generate") {
    generate(data.id, data.chat)
      .then(() => ctx.postMessage({ type: "done", id: data.id }))
      .catch((error: unknown) =>
        ctx.postMessage({
          type: "error",
          id: data.id,
          message: errorMessage(error)
        })
      )
  } else {
    running.get(data.id)?.interrupt()
  }
}
