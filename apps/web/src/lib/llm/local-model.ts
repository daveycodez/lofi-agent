import { EventType } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import type {
  ChatTurn,
  Sampling,
  WorkerRequest,
  WorkerResponse
} from "#/lib/llm/local-model-protocol"
import { createStore } from "#/lib/store"

/**
 * Models that run in the browser with Transformers.js on WebGPU, inside a
 * Web Worker (local-model.worker.ts) so loading and generating never block
 * the page. The worker holds one model at a time.
 */
const SYSTEM_PROMPT =
  "You are a helpful assistant running fully on the user's device. Be concise. Use Markdown when it helps."

export type LocalModelStatus =
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

const statuses = new Map<string, ReturnType<typeof newStatus>>()
const newStatus = () => createStore<LocalModelStatus>({ state: "idle" })

/** Loading status of a model, by Hugging Face id. */
export function localModelStatus(modelId: string) {
  let status = statuses.get(modelId)
  if (!status) {
    status = newStatus()
    statuses.set(modelId, status)
  }
  return status
}

/** Listeners for each running generation, by request id. */
type Generation = {
  onToken: (text: string) => void
  onEnd: (error?: string) => void
}
const generations = new Map<string, Generation>()

type Pending = {
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
}
/** The model the worker holds (or is loading), and its load promise. */
let loaded: { modelId: string; load: Pending } | null = null

let worker: Worker | null = null

function handleMessage({ data }: MessageEvent<WorkerResponse>) {
  if (data.type === "progress") {
    localModelStatus(data.modelId).set({
      state: "loading",
      progress: data.progress,
      cached: data.cached
    })
  } else if (data.type === "ready") {
    localModelStatus(data.modelId).set({ state: "ready" })
    if (loaded?.modelId === data.modelId) loaded.load.resolve()
  } else if (data.type === "load-error") {
    localModelStatus(data.modelId).set({
      state: "error",
      message: data.message
    })
    if (loaded?.modelId === data.modelId) {
      loaded.load.reject(new Error(data.message))
      loaded = null
    }
  } else if (data.type === "token") {
    generations.get(data.id)?.onToken(data.text)
  } else {
    const generation = generations.get(data.id)
    generations.delete(data.id)
    generation?.onEnd(data.type === "error" ? data.message : undefined)
  }
}

function post(message: WorkerRequest) {
  if (!worker) {
    worker = new Worker(new URL("./local-model.worker.ts", import.meta.url), {
      type: "module"
    })
    worker.onmessage = handleMessage
  }
  worker.postMessage(message)
}

/** Loads a model (downloading it the first time), replacing any other. */
export function loadLocalModel(modelId: string) {
  if (loaded?.modelId === modelId) return loaded.load.promise
  if (loaded) {
    // The worker frees the previous model before loading this one.
    localModelStatus(loaded.modelId).set({ state: "idle" })
    loaded.load.reject(new Error("Switched to another model."))
  }
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((res, rej) => {
    resolve = res
    reject = rej
  })
  promise.catch(() => undefined)
  loaded = { modelId, load: { promise, resolve, reject } }
  // The worker reports whether this is a download or a cached load.
  localModelStatus(modelId).set({ state: "loading", progress: 0, cached: true })
  post({ type: "load", modelId })
  return promise
}

function textOf(message: UIMessage | ModelMessage) {
  if ("parts" in message) {
    return message.parts
      .map((p) => (p.type === "text" ? p.content : ""))
      .join("")
  }
  return typeof message.content === "string" ? message.content : ""
}

/** A TanStack AI connection that runs `modelId` locally. */
export function localModelConnection(options: {
  modelId: string
  name: string
  /** The model card's recommended sampling. */
  sampling: Sampling
  /** Whether to think before answering, read when each message is sent. */
  thinking?: () => boolean
}): ConnectConnectionAdapter {
  const { modelId, name, sampling, thinking = () => false } = options
  return {
    async *connect(messages, _data, abortSignal, runContext) {
      const { threadId, runId } = runContext
      const messageId = `msg_${runId}`
      yield { type: EventType.RUN_STARTED, threadId, runId } as StreamChunk

      try {
        await loadLocalModel(modelId)
      } catch (error) {
        yield {
          type: EventType.RUN_ERROR,
          runId,
          message: `Couldn't load ${name}: ${error instanceof Error ? error.message : String(error)}`
        } as StreamChunk
        return
      }

      const chat: Array<ChatTurn> = [{ role: "system", content: SYSTEM_PROMPT }]
      for (const m of messages) {
        if (m.role !== "user" && m.role !== "assistant") continue
        const content = textOf(m)
        if (content) chat.push({ role: m.role, content })
      }

      // Bridge the worker's messages to this async generator.
      const queue: Array<string> = []
      let wake: (() => void) | null = null
      let finished = false
      let failure: string | undefined
      const nextChunk = () =>
        new Promise<void>((resolve) => {
          wake = resolve
        })
      generations.set(runId, {
        onToken: (text) => {
          queue.push(text)
          wake?.()
        },
        onEnd: (error) => {
          failure = error
          finished = true
          wake?.()
        }
      })
      const onAbort = () => post({ type: "stop", id: runId })
      abortSignal?.addEventListener("abort", onAbort)
      const think = thinking()
      // The chat template opens the <think> block in the prompt itself, so
      // add it to the reply too: the thread then shows it as thinking.
      if (think) queue.push("<think>")
      post({ type: "generate", id: runId, chat, thinking: think, sampling })

      yield {
        type: EventType.TEXT_MESSAGE_START,
        messageId,
        role: "assistant"
      } as StreamChunk
      while (!finished || queue.length > 0) {
        const delta = queue.shift()
        if (delta !== undefined) {
          yield {
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta
          } as StreamChunk
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
          message: `${name} failed: ${failure}`
        } as StreamChunk
        return
      }
      yield {
        type: EventType.RUN_FINISHED,
        threadId,
        runId,
        finishReason: "stop",
        model: modelId
      } as StreamChunk
    }
  }
}
