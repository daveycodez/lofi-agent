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
import type { ParsedToolCall, ToolFormat } from "#/lib/llm/tool-calls"
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
  const all = convertMessagesToModelMessages(messages)
  // Tool calls and results from earlier turns are left out; their answers
  // stay. Small models copy the shape of past calls badly and get lost in
  // old results; the current turn's calls and results are kept, since the
  // model answers from them.
  const currentTurn = all.map((m) => m.role).lastIndexOf("user")
  for (const [i, m] of all.entries()) {
    const content = contentText(m.content)
    const earlier = i < currentTurn
    if (m.role === "tool") {
      if (!earlier) chat.push({ role: "tool", content: toolResultText(m) })
      continue
    }
    const calls = (m.toolCalls ?? []).map((call) => ({
      name: call.function.name,
      args: parseArguments(call.function.arguments)
    }))
    if (m.role === "user" || calls.length === 0 || earlier) {
      if (content) chat.push({ role: m.role, content })
      continue
    }
    const turn = assistantToolTurn(content, calls, format)
    // Liquid's guide: keep the reasoning of the current tool-use turn.
    const thinking = m.thinking?.map((t) => t.content).join("")
    chat.push(format === "lfm3" && thinking ? { ...turn, thinking } : turn)
  }
  return chat
}

/** How many times a model is told about calls to tools that don't exist. */
const MAX_UNKNOWN_TOOL_RETRIES = 1

/** An assistant turn with tool calls, written the way the model writes them. */
function assistantToolTurn(
  content: string,
  calls: Array<ParsedToolCall>,
  format: ToolFormat
): ChatTurn {
  if (format === "lfm2") {
    return { role: "assistant", content: content + formatLfm2ToolCalls(calls) }
  }
  return {
    role: "assistant",
    content,
    tool_calls: calls.map((c) => ({
      type: "function",
      function: { name: c.name, arguments: c.args }
    }))
  }
}

/**
 * Runs one generation in the worker and streams it as AG-UI events: a
 * leading <think> block as reasoning, then the answer as text up to any tool
 * call. Returns what's left: the last of the text, the tool-call markup (held
 * back to parse), and the error if generation failed.
 */
async function* generateOnce(options: {
  id: string
  chat: Array<ChatTurn>
  tools: Array<ToolSpec>
  thinking: boolean
  sampling: Sampling
  messageId: string
  abortSignal?: AbortSignal
}): AsyncGenerator<
  StreamChunk,
  { text: string; markup: string | null; failure?: string }
> {
  const { id, messageId, abortSignal } = options

  // Bridge the worker's messages to this async generator.
  const queue: Array<string> = []
  let wake: (() => void) | null = null
  let finished = false
  let failure: string | undefined
  const nextChunk = () =>
    new Promise<void>((resolve) => {
      wake = resolve
    })
  generations.set(id, {
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
  const onAbort = () => post({ type: "stop", id })
  abortSignal?.addEventListener("abort", onAbort)
  post({
    type: "generate",
    id,
    chat: options.chat,
    tools: options.tools,
    thinking: options.thinking,
    sampling: options.sampling
  })

  // Thinking streams as reasoning events, so the message gets a thinking
  // part; the answer streams as its text.
  const reasoningId = `reasoning_${id}`
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
      { type: EventType.REASONING_END, messageId: reasoningId } as StreamChunk
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
      return [...endReasoning(), ...textEvent(toolSplitter.push(piece.text))]
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
  return { ...toolSplitter.flush(), failure }
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

      // The date and time, and the user's system prompt from Settings.
      const chat: Array<ChatTurn> = [
        { role: "system", content: currentSystemPrompt() },
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

      yield {
        type: EventType.TEXT_MESSAGE_START,
        messageId,
        role: "assistant"
      } as StreamChunk
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

      // One generation, streamed into the reply. When the model's only tool
      // calls are to tools that don't exist, it's told so and generates once
      // more, so it answers with what it has instead of stopping there.
      let calls: Array<ParsedToolCall> = []
      let failure: string | undefined
      for (let attempt = 0; ; attempt++) {
        const result = yield* generateOnce({
          id: attempt === 0 ? runId : `${runId}_${attempt}`,
          chat,
          tools,
          thinking: thinking(),
          sampling,
          messageId,
          abortSignal
        })
        failure = result.failure
        yield* textEvent(result.text)
        const parsed = result.markup ? parseToolCalls(result.markup) : []
        calls = parsed.filter((c) => toolNames.has(c.name))
        const unknown = parsed.filter((c) => !toolNames.has(c.name))
        if (
          failure ||
          calls.length > 0 ||
          unknown.length === 0 ||
          attempt >= MAX_UNKNOWN_TOOL_RETRIES ||
          abortSignal?.aborted
        ) {
          // Markup that isn't any tool call is shown as the text it is.
          if (result.markup && parsed.length === 0) {
            yield* textEvent(result.markup)
          }
          break
        }
        chat.push(assistantToolTurn(result.text, unknown, toolFormat))
        for (const call of unknown) {
          chat.push({
            role: "tool",
            content: `Error: there is no tool named "${call.name}". Available tools: ${[...toolNames].join(", ")}. Answer with what you have.`
          })
        }
      }

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
