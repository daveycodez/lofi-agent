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

/** Whether Transformers.js already has the model weights in its browser cache. */
async function isCached() {
  try {
    const cache = await caches.open("transformers-cache")
    const weights = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model_q4f16.onnx_data`
    return (await cache.match(weights)) !== undefined
  } catch {
    return false
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
    const cached = await isCached()
    ctx.postMessage({ type: "progress", progress: 0, cached })
    const tf = (await import(
      /* @vite-ignore */ TRANSFORMERS_URL
    )) as TransformersModule
    const tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL_ID)
    const files: Record<string, [number, number]> = {}
    let lastPercent = -1
    const model = await tf.AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: (p: {
        status: string
        file: string
        loaded?: number
        total?: number
      }) => {
        if (p.status !== "progress" || !p.total) return
        files[p.file] = [p.loaded ?? 0, p.total]
        const [loaded, total] = Object.values(files).reduce(
          (sum, [l, t]) => [sum[0] + l, sum[1] + t],
          [0, 0]
        )
        // Report whole percents only; Transformers.js fires per chunk.
        const percent = Math.floor((loaded / total) * 100)
        if (percent === lastPercent) return
        lastPercent = percent
        ctx.postMessage({ type: "progress", progress: percent / 100, cached })
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
