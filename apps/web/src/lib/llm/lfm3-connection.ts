import { EventType } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import { createStore } from "#/lib/store"

/**
 * LFM3-1.7B running in the browser on WebGPU via Transformers.js.
 * Transformers.js is loaded from the CDN at runtime so its native Node
 * dependencies never enter the app (or Cloudflare) build.
 */
const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2"
const MODEL_ID = "onnx-community/LFM3-1.7B-ONNX"
const SYSTEM_PROMPT =
  "You are a helpful assistant running fully on the user's device. Be concise. Use Markdown when it helps."

export type Lfm3Status =
  | { state: "idle" }
  | {
      state: "loading"
      /** 0-1 across all model files. */
      progress: number
      /** True when the files come from the browser cache, not the network. */
      cached: boolean
    }
  | { state: "ready" }
  | { state: "error"; message: string }

export const lfm3Status = createStore<Lfm3Status>({ state: "idle" })

/* Minimal shapes of the Transformers.js APIs used here. */
type Tensor = { dims: Array<number> }
type Tokenizer = {
  apply_chat_template: (
    messages: Array<{ role: string; content: string }>,
    options: Record<string, unknown>
  ) => string
  (text: string, options: Record<string, unknown>): { input_ids: Tensor }
}
type Model = {
  generate: (options: Record<string, unknown>) => Promise<unknown>
}
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
  InterruptableStoppingCriteria: new () => { interrupt: () => void }
}

let loading: Promise<{
  tf: TransformersModule
  tokenizer: Tokenizer
  model: Model
}> | null = null

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

export function loadLfm3() {
  loading ??= (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("LFM3 needs WebGPU, which this browser doesn't support.")
    }
    const cached = await isCached()
    lfm3Status.set({ state: "loading", progress: 0, cached })
    const tf = (await import(
      /* @vite-ignore */ TRANSFORMERS_URL
    )) as TransformersModule
    const tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL_ID)
    const files: Record<string, [number, number]> = {}
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
        lfm3Status.set({ state: "loading", progress: loaded / total, cached })
      }
    })
    lfm3Status.set({ state: "ready" })
    return { tf, tokenizer, model }
  })().catch((error: unknown) => {
    loading = null
    const message = error instanceof Error ? error.message : String(error)
    lfm3Status.set({ state: "error", message })
    throw error
  })
  return loading
}

function textOf(message: UIMessage | ModelMessage) {
  if ("parts" in message) {
    return message.parts
      .map((p) => (p.type === "text" ? p.content : ""))
      .join("")
  }
  return typeof message.content === "string" ? message.content : ""
}

export const lfm3Connection: ConnectConnectionAdapter = {
  async *connect(messages, _data, abortSignal, runContext) {
    const { threadId, runId } = runContext
    const messageId = `msg_${runId}`
    yield { type: EventType.RUN_STARTED, threadId, runId } as StreamChunk

    let runtime: Awaited<ReturnType<typeof loadLfm3>>
    try {
      runtime = await loadLfm3()
    } catch (error) {
      yield {
        type: EventType.RUN_ERROR,
        runId,
        message: `Couldn't load LFM3: ${error instanceof Error ? error.message : String(error)}`
      } as StreamChunk
      return
    }
    const { tf, tokenizer, model } = runtime

    const chat = [{ role: "system", content: SYSTEM_PROMPT }]
    for (const m of messages) {
      if (m.role !== "user" && m.role !== "assistant") continue
      const content = textOf(m)
      if (content) chat.push({ role: m.role, content })
    }
    // Thinking off: it is slow on small devices and can run out of tokens.
    const prompt = tokenizer.apply_chat_template(chat, {
      add_generation_prompt: true,
      tokenize: false,
      enable_thinking: false
    })
    const inputs = tokenizer(prompt, { add_special_tokens: false })

    // Bridge the streamer callback to this async generator.
    const queue: Array<string> = []
    let wake: (() => void) | null = null
    const nextChunk = () =>
      new Promise<void>((resolve) => {
        wake = resolve
      })
    let finished = false
    let failure: unknown = null
    const stopping = new tf.InterruptableStoppingCriteria()
    const onAbort = () => stopping.interrupt()
    abortSignal?.addEventListener("abort", onAbort)

    const streamer = new tf.TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        queue.push(text)
        wake?.()
      }
    })
    void model
      .generate({
        ...inputs,
        max_new_tokens: 1024,
        do_sample: true,
        temperature: 0.1,
        top_k: 50,
        repetition_penalty: 1.1,
        streamer,
        stopping_criteria: [stopping]
      })
      .catch((error: unknown) => {
        failure = error
      })
      .finally(() => {
        finished = true
        wake?.()
      })

    yield {
      type: EventType.TEXT_MESSAGE_START,
      messageId,
      role: "assistant"
    } as StreamChunk
    while (!finished || queue.length > 0) {
      const delta = queue.shift()
      if (delta !== undefined) {
        if (delta) {
          yield {
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta
          } as StreamChunk
        }
        continue
      }
      await nextChunk()
      wake = null
    }
    abortSignal?.removeEventListener("abort", onAbort)
    yield { type: EventType.TEXT_MESSAGE_END, messageId } as StreamChunk

    if (failure) {
      yield {
        type: EventType.RUN_ERROR,
        runId,
        message: `LFM3 failed: ${failure instanceof Error ? failure.message : String(failure)}`
      } as StreamChunk
      return
    }
    yield {
      type: EventType.RUN_FINISHED,
      threadId,
      runId,
      finishReason: "stop",
      model: MODEL_ID
    } as StreamChunk
  }
}
