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

/** The weights file, under the URL Transformers.js uses as its cache key. */
const WEIGHTS_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model_q4f16.onnx_data`

/**
 * Streams the 1.09 GB weights file from the network straight into
 * Transformers.js's cache, so it's never held in memory during the download.
 * Left to itself, Transformers.js buffers the whole file and then copies it
 * again to cache it, which is enough to crash a tab on iPhone.
 * Returns false when the file was already cached.
 */
async function downloadWeights(onProgress: (progress: number) => void) {
  const cache = await caches.open("transformers-cache")
  if (await cache.match(WEIGHTS_URL)) return false
  // Hugging Face rejects downloads referred from *.workers.dev.
  const res = await fetch(WEIGHTS_URL, { referrerPolicy: "no-referrer" })
  if (!res.ok || !res.body) {
    throw new Error(`Couldn't download the model (HTTP ${res.status}).`)
  }
  const total = Number(res.headers.get("content-length")) || 0
  let loaded = 0
  const counted = res.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        loaded += chunk.byteLength
        if (total) onProgress(loaded / total)
        controller.enqueue(chunk)
      }
    })
  )
  await cache.put(WEIGHTS_URL, new Response(counted, { headers: res.headers }))
  return true
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
    // Download first (whole percents only, the stream reports per chunk),
    // then load from the cache.
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
    const tf = (await import(
      /* @vite-ignore */ TRANSFORMERS_URL
    )) as TransformersModule
    const tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL_ID)
    const model = await tf.AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu"
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
