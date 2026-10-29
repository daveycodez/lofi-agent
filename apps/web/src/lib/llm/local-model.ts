import { EventType, convertMessagesToModelMessages } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import type {
  ChatTurn,
  Sampling,
  ToolSpec,
  WorkerRequest,
  WorkerResponse
} from "#/lib/llm/local-model-protocol"
import {
  createToolCallSplitter,
  formatLfm2ToolCalls,
  parseToolCalls
} from "#/lib/llm/tool-calls"
import type { ToolFormat } from "#/lib/llm/tool-calls"
import { currentSystemPrompt } from "#/lib/settings"
import { createStore } from "#/lib/store"
import { createThinkingSplitter } from "#/lib/thinking"
import type { ReplyPiece } from "#/lib/thinking"

/**
 * Models that run in the browser with Transformers.js on WebGPU, inside a
 * Web Worker (local-model.worker.ts) so loading and generating never block
 * the page. The worker holds one model at a time.
 */
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

/** The text of a message's content (text parts only). */
/**
 * A tool result for the model. A failed tool gives its error message, so the
 * model can see what went wrong and try again.
 */
function toolResultText(m: ModelMessage) {
  const text = contentText(m.content)
  if (m.error) return `Error: ${m.error}`
  return text && text !== "null" ? text : "The tool returned nothing."
}

function contentText(content: ModelMessage["content"]) {
  if (typeof content === "string") return content
  return (content ?? [])
    .map((part) => (part.type === "text" ? part.content : ""))
    .join("")
}

/** Parses a tool call's JSON arguments, or {} if they aren't JSON. */
function parseArguments(json: string): Record<string, unknown> {
  try {
    const args = JSON.parse(json) as unknown
    return args && typeof args === "object"
      ? (args as Record<string, unknown>)
      : {}
  } catch {
    return {}
  }
}

/**
 * The conversation as chat-template turns (the system prompt, if any, is
 * added by the caller). Past tool calls are written the way the model
 * writes them.
 */
function toChatTurns(
  messages: Array<UIMessage> | Array<ModelMessage>,
  format: ToolFormat
) {
  const chat: Array<ChatTurn> = []
  for (const m of convertMessagesToModelMessages(messages)) {
    const content = contentText(m.content)
    if (m.role === "tool") {
      chat.push({ role: "tool", content: toolResultText(m) })
      continue
    }
    const calls = (m.toolCalls ?? []).map((call) => ({
      name: call.function.name,
      args: parseArguments(call.function.arguments)
    }))
    if (m.role === "user" || calls.length === 0) {
      if (content) chat.push({ role: m.role, content })
    } else if (format === "lfm2") {
      chat.push({
        role: "assistant",
        content: content + formatLfm2ToolCalls(calls)
      })
    } else {
      // Liquid's guide: keep the reasoning of the current tool-use turn.
      const thinking = m.thinking?.map((t) => t.content).join("")
      chat.push({
        role: "assistant",
        content,
        ...(thinking && { thinking }),
        tool_calls: calls.map((c) => ({
          type: "function",
          function: { name: c.name, arguments: c.args }
        }))
      })
    }
  }
  return chat
}

/** A TanStack AI connection that runs `modelId` locally. */
export function localModelConnection(options: {
  modelId: string
  name: string
  /** The model card's recommended sampling. */
  sampling: Sampling
  /** How the model writes tool calls. */
  toolFormat: ToolFormat
  /** Whether to think before answering, read when each message is sent. */
  thinking?: () => boolean
}): ConnectConnectionAdapter {
  const {
    modelId,
    name,
    sampling,
    toolFormat,
    thinking = () => false
  } = options
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

      // The user's system prompt from Settings goes first, if they wrote one.
      const systemPrompt = currentSystemPrompt()
      const chat: Array<ChatTurn> = [
        ...(systemPrompt
          ? [{ role: "system" as const, content: systemPrompt }]
          : []),
        ...toChatTurns(messages, toolFormat)
      ]
      // The tools useChat has (from MCP servers), as the template takes them.
      const toolNames = new Set<string>()
      const tools: Array<ToolSpec> = (runContext.clientTools ?? []).map(
        (tool) => {
          toolNames.add(tool.name)
          return {
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.parameters
            }
          }
        }
      )

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
      post({
        type: "generate",
        id: runId,
        chat,
        tools,
        thinking: think,
        sampling
      })

      yield {
        type: EventType.TEXT_MESSAGE_START,
        messageId,
        role: "assistant"
      } as StreamChunk

      // Thinking (a leading <think> block) streams as reasoning events, so
      // the message gets a thinking part; the answer streams as its text,
      // up to any tool call, which is held back and parsed at the end.
      const reasoningId = `reasoning_${runId}`
      let reasoning: "none" | "open" | "closed" = "none"
      const toolSplitter = createToolCallSplitter()
      const textEvent = (delta: string): Array<StreamChunk> =>
        delta
          ? [
              {
                type: EventType.TEXT_MESSAGE_CONTENT,
                messageId,
                delta
              } as StreamChunk
            ]
          : []
      const endReasoning = (): Array<StreamChunk> => {
        if (reasoning !== "open") return []
        reasoning = "closed"
        return [
          {
            type: EventType.REASONING_MESSAGE_END,
            messageId: reasoningId
          } as StreamChunk,
          {
            type: EventType.REASONING_END,
            messageId: reasoningId
          } as StreamChunk
        ]
      }
      const toEvents = (pieces: Array<ReplyPiece>) =>
        pieces.flatMap((piece): Array<StreamChunk> => {
          if (piece.kind === "thinking") {
            const start: Array<StreamChunk> =
              reasoning === "none"
                ? [
                    {
                      type: EventType.REASONING_START,
                      messageId: reasoningId
                    } as StreamChunk,
                    {
                      type: EventType.REASONING_MESSAGE_START,
                      messageId: reasoningId,
                      role: "reasoning"
                    } as StreamChunk
                  ]
                : []
            reasoning = "open"
            return [
              ...start,
              {
                type: EventType.REASONING_MESSAGE_CONTENT,
                messageId: reasoningId,
                delta: piece.text
              } as StreamChunk
            ]
          }
          return [
            ...endReasoning(),
            ...textEvent(toolSplitter.push(piece.text))
          ]
        })

      const splitter = createThinkingSplitter()
      while (!finished || queue.length > 0) {
        const delta = queue.shift()
        if (delta !== undefined) {
          yield* toEvents(splitter.push(delta))
          continue
        }
        await nextChunk()
        wake = null
      }
      abortSignal?.removeEventListener("abort", onAbort)
      yield* toEvents(splitter.flush())
      yield* endReasoning()

      // Tool calls the model made, for tools that exist. Markup that isn't a
      // usable call is shown as text instead.
      const rest = toolSplitter.flush()
      yield* textEvent(rest.text)
      const calls = rest.markup
        ? parseToolCalls(rest.markup).filter((c) => toolNames.has(c.name))
        : []
      if (rest.markup && calls.length === 0) yield* textEvent(rest.markup)
      const callIds = calls.map((_, i) => `call_${runId}_${i}`)
      for (const [i, call] of calls.entries()) {
        const toolCallId = callIds[i]
        yield {
          type: EventType.TOOL_CALL_START,
          toolCallId,
          toolCallName: call.name,
          parentMessageId: messageId
        } as StreamChunk
        yield {
          type: EventType.TOOL_CALL_ARGS,
          toolCallId,
          delta: JSON.stringify(call.args)
        } as StreamChunk
        yield { type: EventType.TOOL_CALL_END, toolCallId } as StreamChunk
      }
      yield { type: EventType.TEXT_MESSAGE_END, messageId } as StreamChunk

      if (failure) {
        yield {
          type: EventType.RUN_ERROR,
          runId,
          message: `${name} failed: ${failure}`
        } as StreamChunk
        return
      }
      // A run that calls browser tools ends with them pending: useChat runs
      // them and sends the results back for the next run.
      yield (
        callIds.length > 0
          ? {
              type: EventType.RUN_FINISHED,
              threadId,
              runId,
              outcome: { type: "success", pendingToolCallIds: callIds },
              model: modelId
            }
          : {
              type: EventType.RUN_FINISHED,
              threadId,
              runId,
              finishReason: "stop",
              model: modelId
            }
      ) as StreamChunk
    }
  }
}
