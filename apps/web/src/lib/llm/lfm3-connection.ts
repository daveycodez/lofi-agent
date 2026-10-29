import { EventType } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import { MODEL_ID } from "#/lib/llm/lfm3-protocol"
import type {
  ChatTurn,
  LoadPause,
  UploadMode,
  WorkerRequest,
  WorkerResponse
} from "#/lib/llm/lfm3-protocol"
import { createStore } from "#/lib/store"

/**
 * LFM3-1.7B running in the browser on WebGPU via Transformers.js, inside a
 * Web Worker (lfm3.worker.ts) so loading and generating never block the page.
 */
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

/** Listeners for each running generation, by request id. */
type Generation = {
  onToken: (text: string) => void
  onEnd: (error?: string) => void
}
const generations = new Map<string, Generation>()

let worker: Worker | null = null
let load: {
  promise: Promise<void>
  resolve: () => void
  reject: (error: Error) => void
} | null = null

function handleMessage({ data }: MessageEvent<WorkerResponse>) {
  if (data.type === "progress") {
    lfm3Status.set({
      state: "loading",
      progress: data.progress,
      cached: data.cached
    })
  } else if (data.type === "ready") {
    lfm3Status.set({ state: "ready" })
    load?.resolve()
  } else if (data.type === "load-error") {
    lfm3Status.set({ state: "error", message: data.message })
    load?.reject(new Error(data.message))
    load = null
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
    worker = new Worker(new URL("./lfm3.worker.ts", import.meta.url), {
      type: "module"
    })
    worker.onmessage = handleMessage
  }
  worker.postMessage(message)
}

/**
 * Experiments: ?upload=write|write-wait picks how weights are sent to the GPU,
 * ?wait=1000&every=32 pauses 1 s after every 32 MB loaded. Read once when the
 * page opens, because sending a message rewrites the URL to ?thread=.
 */
const experimentParams =
  typeof window === "undefined"
    ? new URLSearchParams()
    : new URLSearchParams(window.location.search)
const experiment = (key: string) => experimentParams.get(key)

function uploadMode(): UploadMode {
  const mode = experiment("upload")
  return mode === "write" || mode === "write-wait" ? mode : "default"
}

function loadPause(): LoadPause {
  return {
    ms: Number(experiment("wait")) || 0,
    everyMb: Number(experiment("every")) || 64
  }
}

/** Starts loading the model (downloading it the first time). */
export function loadLfm3() {
  if (!load) {
    let resolve!: () => void
    let reject!: (error: Error) => void
    const promise = new Promise<void>((res, rej) => {
      resolve = res
      reject = rej
    })
    load = { promise, resolve, reject }
    // The worker reports whether this is a download or a cached load.
    lfm3Status.set({ state: "loading", progress: 0, cached: true })
    post({ type: "load", upload: uploadMode(), pause: loadPause() })
  }
  return load.promise
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

    try {
      await loadLfm3()
    } catch (error) {
      yield {
        type: EventType.RUN_ERROR,
        runId,
        message: `Couldn't load LFM3: ${error instanceof Error ? error.message : String(error)}`
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
    post({ type: "generate", id: runId, chat })

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
        message: `LFM3 failed: ${failure}`
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
