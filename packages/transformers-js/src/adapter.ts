/**
 * A TanStack AI text adapter for Transformers.js models that run in the
 * browser, on WebGPU in a Web Worker. Hand it to `chat()` like any hosted
 * model's adapter: `chat()` runs tools and the agent loop, and each
 * `chatStream` call here is one generation in the worker, streamed back as
 * AG-UI events.
 */
import { EventType, normalizeSystemPrompts } from "@tanstack/ai"
import type {
  AdapterYieldChunk,
  ModelMessage,
  TextOptions,
  Tool
} from "@tanstack/ai"
import { BaseTextAdapter } from "@tanstack/ai/adapters"
import type {
  StructuredOutputOptions,
  StructuredOutputResult
} from "@tanstack/ai/adapters"

import { generate, loadModel, setDebug } from "./client"
import type { ChatTurn, RunTimings, Sampling, ToolSpec } from "./protocol"
import { createThinkingSplitter, splitThinking } from "./thinking"
import {
  createToolCallSplitter,
  formatLfm2ToolCalls,
  parseToolCalls
} from "./tool-calls"
import type { ParsedToolCall, ToolFormat } from "./tool-calls"

/** A run's token counts and timings, as the worker measures them. */
export type RunStats = RunTimings

export type TransformersJsOptions = {
  /** How the model writes tool calls: LFM2's markers or LFM3's XML-ish calls. */
  toolFormat: ToolFormat
  /** Sampling from the model card. */
  sampling: Sampling
  /**
   * A separate model to run when reasoning is on, for families whose
   * thinking is its own model (LFM2.5 Thinking), and its sampling.
   */
  thinking?: { model: string; sampling: Sampling }
  /**
   * The run's counts while it generates (a few times a second) and once at
   * the end (`done`). `callId` is unique per model call; `chat()` makes one
   * per step of its agent loop.
   */
  onStats?: (
    stats: RunStats,
    run: { threadId: string; runId: string; callId: string; done: boolean }
  ) => void
  /** Memory and cache logs from the worker, in the console. */
  debug?: boolean
}

type TransformersJsReasoning = {
  levels: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
  budget: false
}

type MetadataByModality = {
  text: unknown
  image: unknown
  audio: unknown
  video: unknown
  document: unknown
}

/** A tool call's text content, or the error a failed tool gave. */
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

/** A tool call's JSON arguments, or {} if they aren't JSON. */
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
 * The conversation as chat-template turns. Tool calls and results from
 * earlier turns are left out and their answers kept: small models copy the
 * shape of past calls badly and get lost in old results. The current turn's
 * calls and results stay, since the model answers from them.
 */
function toChatTurns(messages: Array<ModelMessage>, format: ToolFormat) {
  const chat: Array<ChatTurn> = []
  const currentTurn = messages.map((m) => m.role).lastIndexOf("user")
  for (const [i, m] of messages.entries()) {
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

/** Tools as chat templates take them (OpenAI's function format). */
function toToolSpecs(tools: Array<Tool> | undefined): Array<ToolSpec> {
  return (tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema ?? {
        type: "object",
        properties: {},
        required: []
      }
    }
  }))
}

const usageOf = (stats: RunTimings) => ({
  promptTokens: stats.inputTokens,
  completionTokens: stats.outputTokens,
  totalTokens: stats.inputTokens + stats.outputTokens,
  ...(stats.cachedTokens > 0 && {
    promptTokensDetails: { cachedTokens: stats.cachedTokens }
  })
})

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

let calls = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(calls++).toString(36)}`

export class TransformersJsTextAdapter<
  TModel extends string
> extends BaseTextAdapter<
  TModel,
  Record<string, never>,
  readonly ["text"],
  MetadataByModality,
  ReadonlyArray<string>,
  unknown,
  never,
  TransformersJsReasoning
> {
  readonly name = "transformers-js" as const
  override readonly api = "transformers-js" as const
  private readonly options: TransformersJsOptions

  constructor(model: TModel, options: TransformersJsOptions) {
    super({}, model)
    this.options = options
    if (options.debug) setDebug(true)
  }

  /** The model to run and its sampling, given the reasoning asked for. */
  private pick(reasoning: TextOptions["reasoning"]) {
    const thinks = reasoning !== undefined && reasoning.level !== "off"
    const alternate = thinks ? this.options.thinking : undefined
    return {
      model: alternate?.model ?? this.model,
      sampling: alternate?.sampling ?? this.options.sampling,
      // A separate thinking model always thinks; it takes no switch.
      thinking: thinks
    }
  }

  async *chatStream(options: TextOptions): AsyncIterable<AdapterYieldChunk> {
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const callId = nextId("call")
    const messageId = nextId("msg")
    const reasoningId = nextId("reasoning")
    const signal =
      options.abortController?.signal ??
      (options.request as RequestInit | undefined)?.signal ??
      undefined
    const { model, sampling, thinking } = this.pick(options.reasoning)
    const timestamp = () => Date.now()

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }
    try {
      await loadModel(model)
    } catch (error) {
      yield {
        type: EventType.RUN_ERROR,
        model,
        timestamp: timestamp(),
        message: `Couldn't load ${model}: ${errorText(error)}`,
        error: { message: `Couldn't load ${model}: ${errorText(error)}` }
      }
      return
    }

    const system = normalizeSystemPrompts(options.systemPrompts)
      .map((p) => p.content)
      .join("\n\n")
    const chat: Array<ChatTurn> = [
      ...(system ? [{ role: "system" as const, content: system }] : []),
      ...toChatTurns(options.messages, this.options.toolFormat)
    ]
    const tools = toToolSpecs(options.tools)
    const report = (stats: RunTimings, done: boolean) =>
      this.options.onStats?.(stats, { threadId, runId, callId, done })

    // Thinking streams as reasoning events, then the answer as text up to
    // any tool-call markup, which is held back and parsed at the end.
    let reasoning: "none" | "open" | "closed" = "none"
    let textStarted = false
    let text = ""
    const thinkingSplitter = createThinkingSplitter()
    const toolSplitter = createToolCallSplitter()
    const events: Array<AdapterYieldChunk> = []
    const endReasoning = () => {
      if (reasoning !== "open") return
      reasoning = "closed"
      events.push(
        {
          type: EventType.REASONING_MESSAGE_END,
          messageId: reasoningId,
          model,
          timestamp: timestamp()
        },
        {
          type: EventType.REASONING_END,
          messageId: reasoningId,
          model,
          timestamp: timestamp()
        }
      )
    }
    const addText = (delta: string) => {
      if (!delta) return
      if (!textStarted) {
        textStarted = true
        events.push({
          type: EventType.TEXT_MESSAGE_START,
          messageId,
          role: "assistant",
          model,
          timestamp: timestamp()
        })
      }
      text += delta
      events.push({
        type: EventType.TEXT_MESSAGE_CONTENT,
        messageId,
        delta,
        content: text,
        model,
        timestamp: timestamp()
      })
    }
    const addPieces = (pieces: ReturnType<typeof thinkingSplitter.push>) => {
      for (const piece of pieces) {
        if (piece.kind === "thinking") {
          if (reasoning === "none") {
            events.push(
              {
                type: EventType.REASONING_START,
                messageId: reasoningId,
                model,
                timestamp: timestamp()
              },
              {
                type: EventType.REASONING_MESSAGE_START,
                messageId: reasoningId,
                role: "reasoning",
                model,
                timestamp: timestamp()
              }
            )
          }
          reasoning = "open"
          events.push({
            type: EventType.REASONING_MESSAGE_CONTENT,
            messageId: reasoningId,
            delta: piece.text,
            model,
            timestamp: timestamp()
          })
        } else {
          endReasoning()
          addText(toolSplitter.push(piece.text))
        }
      }
    }

    const stream = generate({
      id: callId,
      chat,
      tools,
      thinking,
      sampling,
      signal,
      onStats: (stats) => report(stats, false)
    })
    let next = await stream.next()
    while (!next.done) {
      addPieces(thinkingSplitter.push(next.value))
      yield* events.splice(0)
      next = await stream.next()
    }
    const { failure, stats } = next.value
    addPieces(thinkingSplitter.flush())
    endReasoning()
    const { text: rest, markup } = toolSplitter.flush()
    addText(rest)
    const parsed = markup ? parseToolCalls(markup) : []
    // Markup that isn't any tool call is shown as the text it is.
    if (markup && parsed.length === 0) addText(markup)
    yield* events.splice(0)
    if (stats) report(stats, true)

    for (const [index, call] of parsed.entries()) {
      const toolCallId = nextId("tool")
      const args = JSON.stringify(call.args)
      yield {
        type: EventType.TOOL_CALL_START,
        toolCallId,
        toolCallName: call.name,
        parentMessageId: messageId,
        index,
        model,
        timestamp: timestamp()
      }
      yield {
        type: EventType.TOOL_CALL_ARGS,
        toolCallId,
        delta: args,
        args,
        model,
        timestamp: timestamp()
      }
      yield {
        type: EventType.TOOL_CALL_END,
        toolCallId,
        toolCallName: call.name,
        args,
        input: call.args,
        model,
        timestamp: timestamp()
      }
    }
    if (textStarted) {
      yield {
        type: EventType.TEXT_MESSAGE_END,
        messageId,
        model,
        timestamp: timestamp()
      }
    }
    if (failure) {
      yield {
        type: EventType.RUN_ERROR,
        model,
        timestamp: timestamp(),
        message: failure,
        error: { message: failure }
      }
      return
    }
    yield {
      type: EventType.RUN_FINISHED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      finishReason: parsed.length > 0 ? "tool_calls" : "stop",
      ...(stats && { usage: usageOf(stats) })
    }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`: the model is asked for JSON
   * matching the schema and its answer parsed. Small models follow schemas
   * loosely; `chat()` validates the result.
   */
  async structuredOutput(
    options: StructuredOutputOptions<Record<string, never>>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const { model, sampling } = this.pick(undefined)
    await loadModel(model)
    const system = normalizeSystemPrompts(chatOptions.systemPrompts)
      .map((p) => p.content)
      .concat(
        `Answer with only a JSON value matching this JSON Schema:\n${JSON.stringify(outputSchema)}`
      )
      .join("\n\n")
    const stream = generate({
      id: nextId("call"),
      chat: [
        { role: "system", content: system },
        ...toChatTurns(chatOptions.messages, this.options.toolFormat)
      ],
      tools: [],
      thinking: false,
      sampling,
      signal: chatOptions.abortController?.signal
    })
    let raw = ""
    let next = await stream.next()
    while (!next.done) {
      raw += next.value
      next = await stream.next()
    }
    if (next.value.failure) throw new Error(next.value.failure)
    const rawText = splitThinking(raw).answer.trim()
    const json = rawText.replace(/^```(?:json)?\s*|\s*```$/g, "")
    try {
      return {
        data: JSON.parse(json) as unknown,
        rawText,
        ...(next.value.stats && { usage: usageOf(next.value.stats) })
      }
    } catch {
      throw new Error(
        `The model's answer isn't JSON: ${rawText.slice(0, 200)}${rawText.length > 200 ? "…" : ""}`
      )
    }
  }
}

/**
 * A TanStack AI adapter for a Transformers.js model in the browser.
 *
 * @example
 * ```ts
 * chat({
 *   adapter: transformersJs("daveycodez/LFM3-1.7B-Web", {
 *     toolFormat: "lfm3",
 *     sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.1 }
 *   }),
 *   messages,
 *   threadId
 * })
 * ```
 */
export function transformersJs<TModel extends string>(
  model: TModel,
  options: TransformersJsOptions
) {
  return new TransformersJsTextAdapter(model, options)
}
