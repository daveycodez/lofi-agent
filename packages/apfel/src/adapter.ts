/**
 * A TanStack AI text adapter for Apple's on-device Foundation Model, served
 * by apfel's OpenAI-compatible server (`apfel --serve --cors`). No backend
 * involved: the browser streams straight from localhost. Hand it to `chat()`
 * like any hosted model's adapter: `chat()` runs tools and the agent loop,
 * and each `chatStream` call here is one completion from apfel, streamed
 * back as AG-UI events.
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

import {
  JSON_TOOL_CALLS_MARKER,
  createToolCallSplitter,
  parseJsonToolCalls
} from "./tool-calls.js"

/** Where `apfel --serve` listens by default. */
export const APFEL_BASE_URL = "http://127.0.0.1:11434/v1"
/** The model id apfel serves Apple's Foundation Model under. */
export const APFEL_MODEL = "apple-foundationmodel"

/**
 * A model call's token counts and timings. apfel reports token counts only
 * in the usage at the end of the stream; counts it doesn't report are left
 * out.
 */
export type ApfelStats = {
  /** Tokens of the prompt. */
  inputTokens?: number
  /** Prompt tokens served from a cache, if apfel reports them. */
  cachedTokens?: number
  /** Tokens the model generated. */
  outputTokens?: number
  /** From sending the request to the first generated token, in ms. */
  firstTokenMs?: number
  /** Tokens generated after the first chunk (estimated, see below). */
  decodeTokens?: number
  /** From the first generated chunk to the last, in ms. */
  decodeMs?: number
}

export type ApfelOptions = {
  /** apfel's OpenAI-compatible API, up to `/v1`. */
  baseUrl?: string
  /** The model id to ask apfel for. */
  model?: string
  /**
   * The call's stats at its first token (`done: false`, time to first token
   * only) and when it ends (`done: true`). `callId` is unique per model
   * call; `chat()` makes one per step of its agent loop.
   */
  onStats?: (
    stats: ApfelStats,
    run: { threadId: string; runId: string; callId: string; done: boolean }
  ) => void
}

/**
 * Request fields apfel takes for Apple's model. apfel rejects parameters
 * the model doesn't support (stop, penalties, n…) with HTTP 400.
 */
export type ApfelModelOptions = {
  temperature?: number
  max_tokens?: number
  seed?: number
}

type ApfelReasoning = never

type MetadataByModality = {
  text: unknown
  image: unknown
  audio: unknown
  video: unknown
  document: unknown
}

type OpenAIToolCall = {
  id: string
  type: "function"
  function: { name: string; arguments: string }
}

type OpenAIMessage =
  | { role: "system" | "user"; content: string }
  | {
      role: "assistant"
      content: string | null
      tool_calls?: Array<OpenAIToolCall>
    }
  | { role: "tool"; tool_call_id: string; content: string }

type OpenAITool = {
  type: "function"
  function: { name: string; description: string; parameters: unknown }
}

type FinishReason = "stop" | "length" | "content_filter" | "tool_calls" | null

type OpenAIUsage = {
  prompt_tokens?: number
  completion_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
}

type OpenAIChunk = {
  choices?: Array<{
    delta?: {
      content?: string | null
      reasoning_content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: FinishReason
  }>
  /** Sent in a last chunk of its own when asked for with include_usage. */
  usage?: OpenAIUsage | null
  error?: { message?: string }
}

type OpenAICompletion = {
  choices?: Array<{
    message?: { content?: string | null; refusal?: string | null }
    finish_reason?: FinishReason
  }>
  usage?: OpenAIUsage | null
}

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

/**
 * The conversation in OpenAI's chat format. Tool calls and results from
 * earlier turns are left out; their answers stay. With them in the history,
 * Apple's model writes its next tool call as broken JSON text most of the
 * time (4 of 12 calls worked, against 12 of 12 without). The current turn's
 * calls and results are kept: the model answers from them.
 */
function toOpenAIMessages(messages: Array<ModelMessage>): Array<OpenAIMessage> {
  const out: Array<OpenAIMessage> = []
  const currentTurn = messages.map((m) => m.role).lastIndexOf("user")
  for (const [i, m] of messages.entries()) {
    const earlierToolStep =
      i < currentTurn &&
      (m.role === "tool" || (m.role === "assistant" && m.toolCalls?.length))
    const content = contentText(m.content)
    if (earlierToolStep) {
      if (m.role === "assistant" && content) {
        out.push({ role: "assistant", content })
      }
    } else if (m.role === "tool") {
      out.push({
        role: "tool",
        tool_call_id: m.toolCallId ?? "",
        content: toolResultText(m)
      })
    } else if (m.role === "assistant" && m.toolCalls?.length) {
      out.push({
        role: "assistant",
        content: content || null,
        tool_calls: m.toolCalls.map((call) => ({
          id: call.id,
          type: "function",
          function: call.function
        }))
      })
    } else if (content) {
      out.push({ role: m.role === "user" ? "user" : "assistant", content })
    }
  }
  return out
}

/** Tools as OpenAI function tools; `inputSchema` is already JSON Schema. */
function toOpenAITools(tools: Array<Tool> | undefined): Array<OpenAITool> {
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

/**
 * `tool_choice` for the request. "auto" is apfel's default and is left out,
 * so a plain request is sent exactly as before.
 */
function toolChoice(choice: TextOptions["toolChoice"]) {
  if (!choice || choice === "auto") return {}
  if (typeof choice === "string") return { tool_choice: choice }
  return {
    tool_choice: { type: "function", function: { name: choice.name } }
  }
}

/** The system prompts as one system message, if there are any. */
function systemMessages(
  prompts: TextOptions["systemPrompts"]
): Array<OpenAIMessage> {
  const system = normalizeSystemPrompts(prompts)
    .map((p) => p.content)
    .join("\n\n")
  return system ? [{ role: "system", content: system }] : []
}

/** Yields the `data:` payloads of an SSE response body. */
async function* sseData(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline: number
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line.startsWith("data:")) yield line.slice(5).trim()
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/** A tool call's JSON arguments, or undefined if they aren't a JSON object. */
function parseArguments(json: string): Record<string, unknown> | undefined {
  try {
    const args = JSON.parse(json) as unknown
    return args && typeof args === "object" && !Array.isArray(args)
      ? (args as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
}

/** The message of apfel's `{"error": {"message"}}` body, or the raw body. */
async function errorBody(response: Response) {
  const raw = await response.text().catch(() => "")
  try {
    const message = (JSON.parse(raw) as OpenAIChunk).error?.message
    if (message) return message
  } catch {
    // Not JSON: the raw body says what it says.
  }
  return raw.slice(0, 200)
}

function usageOf(usage: OpenAIUsage | null | undefined) {
  if (
    usage?.prompt_tokens === undefined ||
    usage.completion_tokens === undefined
  ) {
    return undefined
  }
  const cachedTokens = usage.prompt_tokens_details?.cached_tokens
  return {
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.prompt_tokens + usage.completion_tokens,
    ...(cachedTokens && { promptTokensDetails: { cachedTokens } })
  }
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

const unreachable = (baseUrl: string) =>
  `Couldn't reach Apple's on-device model at ${baseUrl}. Start it with \`apfel --serve --cors\`.`

let ids = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(ids++).toString(36)}`

type StreamedCall = { name: string; arguments: string }

export class ApfelTextAdapter extends BaseTextAdapter<
  string,
  ApfelModelOptions,
  readonly ["text"],
  MetadataByModality,
  ReadonlyArray<string>,
  unknown,
  never,
  ApfelReasoning
> {
  readonly name = "apfel" as const
  override readonly api = "apfel" as const
  private readonly baseUrl: string
  private readonly onStats: ApfelOptions["onStats"]

  constructor(options: ApfelOptions = {}) {
    super({}, options.model ?? APFEL_MODEL)
    this.baseUrl = (options.baseUrl ?? APFEL_BASE_URL).replace(/\/+$/, "")
    this.onStats = options.onStats
  }

  async *chatStream(
    options: TextOptions<ApfelModelOptions>
  ): AsyncIterable<AdapterYieldChunk> {
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const callId = nextId("call")
    const messageId = nextId("msg")
    const reasoningId = nextId("reasoning")
    const model = this.model
    const { logger } = options
    const signal =
      options.abortController?.signal ??
      (options.request as RequestInit | undefined)?.signal ??
      undefined
    const timestamp = () => Date.now()
    const runError = (message: string): AdapterYieldChunk => ({
      type: EventType.RUN_ERROR,
      model,
      timestamp: timestamp(),
      message,
      error: { message }
    })

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }

    const tools = toOpenAITools(options.tools)
    const toolNames = new Set(tools.map((tool) => tool.function.name))
    const send = options.wrapFetch ? options.wrapFetch(fetch) : fetch
    const url = `${this.baseUrl}/chat/completions`
    logger.request(
      `activity=chat provider=apfel model=${model} messages=${options.messages.length} tools=${tools.length} stream=true`,
      { provider: "apfel", model }
    )
    const sentAt = performance.now()
    let response: Response
    try {
      response = await send(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...options.modelOptions,
          model,
          stream: true,
          // Token counts, for the stats.
          stream_options: { include_usage: true },
          messages: [
            ...systemMessages(options.systemPrompts),
            ...toOpenAIMessages(options.messages)
          ],
          ...(tools.length > 0 && {
            tools,
            ...toolChoice(options.toolChoice)
          })
        }),
        signal
      })
    } catch (error) {
      if (signal?.aborted) return
      logger.errors("apfel.chatStream fetch failed", {
        error,
        source: "apfel.chatStream"
      })
      yield runError(unreachable(this.baseUrl))
      return
    }
    if (!response.ok || !response.body) {
      yield runError(
        `apfel returned HTTP ${response.status}: ${await errorBody(response)}`
      )
      return
    }

    // Reasoning (apfel sends none for Apple's model today) streams as
    // reasoning events, then the answer as text up to any tool call the
    // model wrote as text, which is held back and parsed at the end.
    let reasoning: "none" | "open" | "closed" = "none"
    let textStarted = false
    let text = ""
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
    const addReasoning = (delta: string) => {
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
        delta,
        model,
        timestamp: timestamp()
      })
    }
    const addText = (delta: string) => {
      if (!delta) return
      endReasoning()
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
    const endText = () => {
      if (!textStarted) return
      textStarted = false
      events.push({
        type: EventType.TEXT_MESSAGE_END,
        messageId,
        model,
        timestamp: timestamp()
      })
    }

    // When apfel doesn't recognize a tool call, it passes it on as text;
    // that part is held back and read once the reply ends.
    const splitter = createToolCallSplitter(JSON_TOOL_CALLS_MARKER)
    // Tool calls stream in pieces, by index.
    const streamed: Array<StreamedCall> = []
    // apfel streams a few tokens per chunk: the first chunk's share of the
    // characters stands in for its share of the tokens.
    const timing = { first: 0, last: 0, firstChars: 0, chars: 0 }
    let usage: OpenAIChunk["usage"]
    let finishReason: FinishReason = null
    try {
      for await (const data of sseData(response.body)) {
        if (data === "[DONE]") break
        const chunk = JSON.parse(data) as OpenAIChunk
        logger.provider("provider=apfel", { chunk })
        if (chunk.error) {
          throw new Error(chunk.error.message ?? "apfel sent an error.")
        }
        usage = chunk.usage ?? usage
        const choice = chunk.choices?.[0]
        const delta = choice?.delta
        const chars =
          (delta?.content?.length ?? 0) +
          (delta?.reasoning_content?.length ?? 0) +
          (delta?.tool_calls ?? []).reduce(
            (n, piece) =>
              n +
              (piece.function?.name?.length ?? 0) +
              (piece.function?.arguments?.length ?? 0),
            0
          )
        if (chars > 0) {
          timing.last = performance.now()
          if (!timing.first) {
            timing.first = timing.last
            timing.firstChars = chars
            // Token counts come only with the usage at the end.
            this.onStats?.(
              { firstTokenMs: timing.first - sentAt },
              { threadId, runId, callId, done: false }
            )
          }
          timing.chars += chars
        }
        if (delta?.reasoning_content) addReasoning(delta.reasoning_content)
        if (delta?.content) addText(splitter.push(delta.content))
        for (const piece of delta?.tool_calls ?? []) {
          // apfel numbers its calls call_1, call_2… in every response; the
          // adapter gives them ids of its own that are unique in the thread.
          const call = (streamed[piece.index] ??= { name: "", arguments: "" })
          call.name += piece.function?.name ?? ""
          call.arguments += piece.function?.arguments ?? ""
        }
        if (choice?.finish_reason) finishReason = choice.finish_reason
        yield* events.splice(0)
      }
    } catch (error) {
      if (signal?.aborted) return
      logger.errors("apfel.chatStream stream failed", {
        error,
        source: "apfel.chatStream"
      })
      endReasoning()
      endText()
      yield* events.splice(0)
      yield runError(`apfel's stream failed: ${errorText(error)}`)
      return
    }

    const outputTokens = usage?.completion_tokens
    this.onStats?.(
      {
        inputTokens: usage?.prompt_tokens,
        cachedTokens: usage?.prompt_tokens_details?.cached_tokens,
        outputTokens,
        ...(timing.first && {
          firstTokenMs: timing.first - sentAt,
          ...(outputTokens !== undefined && {
            decodeTokens:
              (outputTokens * (timing.chars - timing.firstChars)) /
              timing.chars,
            decodeMs: timing.last - timing.first
          })
        })
      },
      { threadId, runId, callId, done: true }
    )

    endReasoning()
    const { text: rest, markup } = splitter.flush()
    const fromText =
      markup && streamed.length === 0
        ? parseJsonToolCalls(markup, toolNames)
        : []
    // Text that wasn't a readable tool call is shown after all.
    addText(`${rest}${fromText.length > 0 ? "" : (markup ?? "")}`)
    yield* events.splice(0)

    // Calls to tools that don't exist are passed on too: `chat()` answers
    // them with an "Unknown tool" error, so the model can answer with what
    // it has instead of stopping there.
    const calls = [
      ...streamed.filter((call) => call?.name),
      ...fromText.map((call) => ({
        name: call.name,
        arguments: JSON.stringify(call.args)
      }))
    ]
    for (const [index, call] of calls.entries()) {
      const toolCallId = nextId("tool")
      const args = call.arguments || "{}"
      const input = parseArguments(args)
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
        // Arguments that aren't JSON are left out; `chat()` reports them.
        ...(input && { input }),
        model,
        timestamp: timestamp()
      }
    }
    endText()
    yield* events.splice(0)

    const finishUsage = usageOf(usage)
    yield {
      type: EventType.RUN_FINISHED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      finishReason:
        calls.length > 0
          ? "tool_calls"
          : finishReason === "tool_calls" || finishReason === null
            ? "stop"
            : finishReason,
      ...(finishUsage && { usage: finishUsage })
    }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`, from apfel's JSON mode:
   * `response_format: json_schema` constrains Apple's model to the schema
   * (guided generation). `chat()` validates the result.
   */
  async structuredOutput(
    options: StructuredOutputOptions<ApfelModelOptions>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const model = this.model
    const send = chatOptions.wrapFetch ? chatOptions.wrapFetch(fetch) : fetch
    chatOptions.logger.request(
      `activity=chat provider=apfel model=${model} messages=${chatOptions.messages.length} tools=0 stream=false`,
      { provider: "apfel", model }
    )
    let response: Response
    try {
      response = await send(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...chatOptions.modelOptions,
          model,
          messages: [
            ...systemMessages(chatOptions.systemPrompts),
            ...toOpenAIMessages(chatOptions.messages)
          ],
          response_format: {
            type: "json_schema",
            json_schema: { name: "output", schema: outputSchema }
          }
        }),
        signal:
          chatOptions.abortController?.signal ??
          (chatOptions.request as RequestInit | undefined)?.signal ??
          undefined
      })
    } catch (error) {
      chatOptions.logger.errors("apfel.structuredOutput fetch failed", {
        error,
        source: "apfel.structuredOutput"
      })
      if (chatOptions.abortController?.signal.aborted) throw error
      throw new Error(unreachable(this.baseUrl))
    }
    if (!response.ok) {
      throw new Error(
        `apfel returned HTTP ${response.status}: ${await errorBody(response)}`
      )
    }
    const completion = (await response.json()) as OpenAICompletion
    const choice = completion.choices?.[0]
    if (choice?.finish_reason === "length") {
      throw new Error(
        "apfel's answer was cut off at the token limit before its JSON ended; raise modelOptions.max_tokens."
      )
    }
    const rawText = (
      choice?.message?.content ??
      choice?.message?.refusal ??
      ""
    ).trim()
    const json = rawText.replace(/^```(?:json)?\s*|\s*```$/g, "")
    const usage = usageOf(completion.usage)
    try {
      return {
        data: JSON.parse(json) as unknown,
        rawText,
        model,
        ...(usage && { usage })
      }
    } catch {
      throw new Error(
        `The model's answer isn't JSON: ${rawText.slice(0, 200)}${rawText.length > 200 ? "…" : ""}`
      )
    }
  }
}

/**
 * A TanStack AI adapter for Apple's on-device Foundation Model through
 * apfel (`apfel --serve --cors`).
 *
 * @example
 * ```ts
 * chat({ adapter: apfel(), messages, threadId })
 * ```
 */
export function apfel(options?: ApfelOptions) {
  return new ApfelTextAdapter(options)
}

/**
 * Whether apfel is running and serves the model: true if `GET /v1/models`
 * lists it within 10 seconds.
 */
export async function isApfelAvailable(
  options: Pick<ApfelOptions, "baseUrl" | "model"> = {}
) {
  const baseUrl = (options.baseUrl ?? APFEL_BASE_URL).replace(/\/+$/, "")
  const model = options.model ?? APFEL_MODEL
  try {
    const response = await fetch(`${baseUrl}/models`, {
      signal: AbortSignal.timeout(10_000)
    })
    if (!response.ok) return false
    const list = (await response.json()) as { data?: Array<{ id?: string }> }
    return list.data?.some((m) => m.id === model) ?? false
  } catch {
    return false
  }
}
