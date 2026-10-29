/**
 * Runs a local model with Transformers.js on WebGPU, off the main thread so
 * the page stays smooth while it loads and generates. One model at a time:
 * loading another frees the previous one. Transformers.js comes from the CDN
 * at runtime so its native Node dependencies never enter the app build.
 */
import type {
  ChatTurn,
  WorkerRequest,
  WorkerResponse
} from "./local-model-protocol"

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
  dispose: () => Promise<unknown>
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
type Runtime = { tf: TransformersModule; tokenizer: Tokenizer; model: Model }

const ctx = globalThis as unknown as {
  postMessage: (message: WorkerResponse) => void
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
}

/** Whether the model's weights are already in Transformers.js's cache. */
async function isCached(modelId: string) {
  try {
    const cache = await caches.open("transformers-cache")
    const weights = `https://huggingface.co/${modelId}/resolve/main/onnx/model_q4f16.onnx_data`
    return (await cache.match(weights)) !== undefined
  } catch {
    return false
  }
}

/** Removes model files an earlier version of the app kept in OPFS. */
async function removeOldOpfsFiles() {
  try {
    const dir = await navigator.storage.getDirectory()
    for await (const name of (
      dir as unknown as { keys: () => AsyncIterable<string> }
    ).keys()) {
      if (name.startsWith("lfm3-1.7b-")) await dir.removeEntry(name)
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
  const cached = await isCached(modelId)
  ctx.postMessage({ type: "progress", modelId, progress: 0, cached })
  const tf = (await import(
    /* @vite-ignore */ TRANSFORMERS_URL
  )) as TransformersModule
  const tokenizer = await tf.AutoTokenizer.from_pretrained(modelId)
  const files: Record<string, [number, number]> = {}
  let lastPercent = -1
  const model = await tf.AutoModelForCausalLM.from_pretrained(modelId, {
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
      // Report whole percents only; Transformers.js reports every chunk.
      const percent = Math.floor((loaded / total) * 100)
      if (percent === lastPercent) return
      lastPercent = percent
      ctx.postMessage({
        type: "progress",
        modelId,
        progress: percent / 100,
        cached
      })
    }
  })
  return { tf, tokenizer, model }
}

/** Loads `modelId`, freeing whichever model was loaded before. */
function load(modelId: string) {
  if (current?.modelId === modelId) return current.runtime
  const previous = current?.runtime
  const runtime = (async () => {
    await previous?.then((r) => r.model.dispose()).catch(() => undefined)
    void removeOldOpfsFiles()
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

async function generate(id: string, chat: Array<ChatTurn>, thinking: boolean) {
  if (!current) throw new Error("No model loaded.")
  const { tf, tokenizer, model } = await current.runtime
  // Models with a thinking mode reason in a <think> block before answering.
  const prompt = tokenizer.apply_chat_template(chat, {
    add_generation_prompt: true,
    tokenize: false,
    enable_thinking: thinking
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
    // Liquid's recommended sampling for LFM models.
    await model.generate({
      ...inputs,
      // Thinking takes many more tokens before the answer starts.
      max_new_tokens: thinking ? 4096 : 1024,
      do_sample: true,
      temperature: 0.1,
      top_k: 50,
      repetition_penalty: 1.05,
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
    load(data.modelId).catch(() => undefined)
  } else if (data.type === "generate") {
    generate(data.id, data.chat, data.thinking)
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
