/**
 * A TanStack AI text adapter for GGUF models that run in the browser on the
 * CPU, with wllama (llama.cpp in WebAssembly). Hand it to `chat()` like any
 * hosted model's adapter: `chat()` runs tools and the agent loop, and each
 * `chatStream` call here is one generation, streamed back as AG-UI events.
 */
import { buildBaseUsage, EventType, normalizeSystemPrompts } from "@tanstack/ai"
import type {
  AdapterYieldChunk,
  ModelMessage,
  TextOptions,
  TokenUsage,
  Tool
} from "@tanstack/ai"
import { BaseTextAdapter } from "@tanstack/ai/adapters"
import type {
  StructuredOutputOptions,
  StructuredOutputResult
} from "@tanstack/ai/adapters"

import { configureRuntime, generate, loadModel } from "./client.js"
import type { RunTimings } from "./client.js"
import { modelMeta, WLLAMA_MODEL_REASONING } from "./model-meta.js"
import type {
  Sampling,
  WllamaModelReasoningByName,
  WllamaTextModel
} from "./model-meta.js"
import { renderChat } from "./template.js"
import type { ChatTurn, ToolSpec } from "./template.js"
import { createThinkingSplitter, splitThinking } from "./thinking.js"
import {
  createToolCallSplitter,
  formatLfm2ToolCalls,
  parseToolCalls,
  TOOL_CALL_END,
  TOOL_CALL_START
} from "./tool-calls.js"
import type { ParsedToolCall } from "./tool-calls.js"

export type WllamaTextConfig = {
  /**
   * Where wllama's WebAssembly file is served, for bundlers that don't
   * emit it from `new URL(…, import.meta.url)`. Applies to every model.
   */
  wasmUrl?: string
  /**
   * Where wllama's compat build (for Safari) is served, if not from
   * `@wllama/wllama-compat` through the bundler.
   */
  compat?: { wasm: string; worker: string }
  /** CPU threads: three quarters of the cores where the page is cross-origin isolated, else one. */
  threads?: number
  /** wllama's and llama.cpp's logs, in the console. */
  debug?: boolean
}

/** Per-call `modelOptions`: sampling over the model card's, and the most tokens to generate. */
export type WllamaTextProviderOptions = Partial<Sampling> & {
  /** Most tokens to generate: 4,096 for a model that thinks, else 1,024. */
  max_new_tokens?: number
}

/**
 * Timings of one model call, in milliseconds, in `RUN_FINISHED`'s
 * `usage.providerUsageDetails`.
 */
export type WllamaProviderUsageDetails = {
  /** From the request to the first generated token, reading the prompt included. */
  firstTokenMs?: number
  /** Reading the part of the prompt that wasn't cached. */
  promptMs?: number
  /** Generating. */
  decodeMs?: number
  /** Tokens generated over `decodeMs`. */
  decodeTokens?: number
}

/** The reasoning levels of a model, for `chat({ reasoning })`; `never` for one that doesn't reason. */
type ResolveReasoning<TModel extends string> =
  TModel extends keyof WllamaModelReasoningByName
    ? WllamaModelReasoningByName[TModel]
    : never

type MetadataByModality = {
  text: unknown
  image: unknown
  audio: unknown
  video: unknown
  document: unknown
}

/** Where the model's thinking starts and ends. */
const THINK = { open: "<think>", close: "</think>" }

/**
 * Each reply as the model wrote it (thinking and tool-call markup
 * included), by what the conversation keeps of it: its tool calls' ids, or
 * its answer. Written back into the next prompt as it was generated, the
 * prompt starts with exactly what llama.cpp has read, so it reads only
 * what's new.
 */
const replies = new Map<string, string>()
const MAX_REPLIES = 256

const replyKey = (content: string, toolCallIds: Array<string>) =>
  toolCallIds.length > 0
    ? `tools:${toolCallIds.join(",")}`
    : `text:${content.trim()}`

function rememberReply(key: string, raw: string) {
  replies.delete(key)
  replies.set(key, raw)
  if (replies.size > MAX_REPLIES) {
    replies.delete(replies.keys().next().value as string)
  }
}

function contentText(content: ModelMessage["content"]) {
  if (typeof content === "string") return content
  return (content ?? [])
    .map((part) => (part.type === "text" ? part.content : ""))
    .join("")
}

/** A tool call's text content, or the error a failed tool gave. */
function toolResultText(m: ModelMessage) {
  const text = contentText(m.content)
  if (m.error) return `Error: ${m.error}`
  return text && text !== "null" ? text : "The tool returned nothing."
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

/**
 * The conversation as chat-template turns. Tool calls and results from
 * earlier turns are left out and their answers kept: small models copy the
 * shape of past calls badly and get lost in old results. The current turn's
 * calls and results stay, since the model answers from them. Replies are
 * written as the model wrote them, else rebuilt from their parts.
 */
function toChatTurns(messages: Array<ModelMessage>): Array<ChatTurn> {
  const chat: Array<ChatTurn> = []
  const currentTurn = messages.map((m) => m.role).lastIndexOf("user")
  for (const [i, m] of messages.entries()) {
    const content = contentText(m.content)
    const earlier = i < currentTurn
    if (m.role === "tool") {
      if (!earlier) chat.push({ role: "tool", content: toolResultText(m) })
      continue
    }
    if (m.role !== "assistant") {
      if (content) chat.push({ role: m.role, content })
      continue
    }
    const calls = m.toolCalls ?? []
    if (calls.length > 0 && earlier) {
      if (content) chat.push({ role: "assistant", content })
      continue
    }
    const raw = replies.get(
      replyKey(
        content,
        calls.map((c) => c.id)
      )
    )
    if (raw !== undefined) {
      chat.push({ role: "assistant", content: raw })
      continue
    }
    const thinking = m.thinking?.map((t) => t.content).join("") ?? ""
    const parsed: Array<ParsedToolCall> = calls.map((c) => ({
      name: c.function.name,
      args: parseArguments(c.function.arguments)
    }))
    const text =
      (thinking ? `${THINK.open}${thinking}${THINK.close}` : "") +
      content +
      (parsed.length > 0 ? formatLfm2ToolCalls(parsed) : "")
    if (text) chat.push({ role: "assistant", content: text })
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

/** A model call's counts as TanStack AI usage, its timings as provider details. */
function usageOf(stats: RunTimings): TokenUsage {
  const usage = buildBaseUsage<WllamaProviderUsageDetails>({
    promptTokens: stats.inputTokens,
    completionTokens: stats.outputTokens,
    totalTokens: stats.inputTokens + stats.outputTokens
  })
  if (stats.cachedTokens > 0) {
    usage.promptTokensDetails = { cachedTokens: stats.cachedTokens }
  }
  usage.providerUsageDetails = {
    ...(stats.firstTokenMs !== null && { firstTokenMs: stats.firstTokenMs }),
    ...(stats.promptMs !== null && { promptMs: stats.promptMs }),
    ...(stats.decodeTokens > 0 && {
      decodeMs: stats.decodeMs,
      decodeTokens: stats.decodeTokens
    })
  }
  return usage
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

let calls = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(calls++).toString(36)}`

export class WllamaTextAdapter<
  TModel extends WllamaTextModel
> extends BaseTextAdapter<
  TModel,
  WllamaTextProviderOptions,
  readonly ["text"],
  MetadataByModality,
  ReadonlyArray<string>,
  unknown,
  never,
  ResolveReasoning<TModel>
> {
  override readonly kind = "text" as const
  readonly name = "wllama" as const
  override readonly api = "wllama" as const
  override readonly inputModalities = ["text"] as const

  constructor(config: WllamaTextConfig, model: TModel) {
    super({}, model)
    configureRuntime({
      ...(config.wasmUrl && { wasmUrl: config.wasmUrl }),
      ...(config.compat && { compat: config.compat }),
      ...(config.threads && { threads: config.threads }),
      ...(config.debug && { debug: true })
    })
  }

  /** Sampling for a call: `modelOptions` over the model card's. */
  private sampling(modelOptions: WllamaTextProviderOptions | undefined) {
    const base = modelMeta(this.model).sampling
    const top_p = modelOptions?.top_p ?? base.top_p
    return {
      temperature: modelOptions?.temperature ?? base.temperature,
      top_k: modelOptions?.top_k ?? base.top_k,
      repeat_penalty: modelOptions?.repeat_penalty ?? base.repeat_penalty,
      ...(top_p !== undefined && { top_p })
    }
  }

  /** Most tokens to generate: thinking takes many before the answer starts. */
  private maxTokens(modelOptions: WllamaTextProviderOptions | undefined) {
    return (
      modelOptions?.max_new_tokens ??
      (WLLAMA_MODEL_REASONING[this.model] ? 4096 : 1024)
    )
  }

  /** The prompt for a call, written by the model's chat template. */
  private async prompt(
    systemPrompts: TextOptions["systemPrompts"],
    messages: Array<ModelMessage>,
    tools: Array<ToolSpec>,
    extraSystem?: string
  ) {
    const wllama = await loadModel(this.model)
    const template = wllama.getChatTemplate()
    if (!template) throw new Error(`${this.model} has no chat template.`)
    const system = normalizeSystemPrompts(systemPrompts)
      .map((p) => p.content)
      .concat(extraSystem ?? [])
      .join("\n\n")
    const chat: Array<ChatTurn> = [
      ...(system ? [{ role: "system" as const, content: system }] : []),
      ...toChatTurns(messages)
    ]
    return renderChat(template, chat, tools)
  }

  async *chatStream(
    options: TextOptions<WllamaTextProviderOptions>
  ): AsyncIterable<AdapterYieldChunk> {
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const messageId = nextId("msg")
    const reasoningId = nextId("reasoning")
    const signal =
      options.abortController?.signal ??
      (options.request as RequestInit | undefined)?.signal ??
      undefined
    const model = this.model
    const timestamp = () => Date.now()

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }
    let prompt: string
    try {
      prompt = await this.prompt(
        options.systemPrompts,
        options.messages,
        toToolSpecs(options.tools)
      )
    } catch (error) {
      const message = `Couldn't load ${model}: ${errorText(error)}`
      yield {
        type: EventType.RUN_ERROR,
        model,
        timestamp: timestamp(),
        message,
        error: { message }
      }
      return
    }

    // Thinking streams as reasoning events, then the answer as text up to
    // any tool-call markup, which is held back and parsed at the end.
    let reasoning: "none" | "open" | "closed" = "none"
    let textStarted = false
    let text = ""
    let raw = ""
    const thinkingSplitter = createThinkingSplitter(THINK)
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

    let failure: string | undefined
    let stats: RunTimings | undefined
    try {
      const stream = generate({
        model,
        prompt,
        sampling: this.sampling(options.modelOptions),
        maxTokens: this.maxTokens(options.modelOptions),
        // A reply ends with its tool calls: what LFM2.5 writes after them
        // isn't shown, and the next step starts from their results.
        stop: [TOOL_CALL_END],
        signal
      })
      let next = await stream.next()
      while (!next.done) {
        raw += next.value
        addPieces(thinkingSplitter.push(next.value))
        yield* events.splice(0)
        next = await stream.next()
      }
      stats = next.value
    } catch (error) {
      failure = errorText(error)
    }
    // It stopped at the end of its tool calls, which llama.cpp leaves out.
    if (raw.includes(TOOL_CALL_START) && !raw.endsWith(TOOL_CALL_END)) {
      if (!stats?.truncated) raw += TOOL_CALL_END
    }
    addPieces(thinkingSplitter.flush())
    endReasoning()
    const { text: rest, markup } = toolSplitter.flush()
    addText(rest)
    const parsed = markup ? parseToolCalls(markup) : []
    // Markup that isn't any tool call is shown as the text it is.
    if (markup && parsed.length === 0) addText(markup)
    yield* events.splice(0)

    const toolCallIds = parsed.map(() => nextId("tool"))
    if (!failure && !signal?.aborted) {
      rememberReply(replyKey(text, toolCallIds), raw)
    }
    for (const [index, call] of parsed.entries()) {
      const toolCallId = toolCallIds[index]
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
      finishReason:
        parsed.length > 0 ? "tool_calls" : stats?.truncated ? "length" : "stop",
      ...(stats && { usage: usageOf(stats) })
    }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`: the model is asked for JSON
   * matching the schema and its answer parsed. Small models follow schemas
   * loosely; `chat()` validates the result.
   */
  async structuredOutput(
    options: StructuredOutputOptions<WllamaTextProviderOptions>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const prompt = await this.prompt(
      chatOptions.systemPrompts,
      chatOptions.messages,
      [],
      `Answer with only a JSON value matching this JSON Schema:\n${JSON.stringify(outputSchema)}`
    )
    const stream = generate({
      model: this.model,
      prompt,
      sampling: this.sampling(chatOptions.modelOptions),
      maxTokens: this.maxTokens(chatOptions.modelOptions),
      signal: chatOptions.abortController?.signal
    })
    let raw = ""
    let next = await stream.next()
    while (!next.done) {
      raw += next.value
      next = await stream.next()
    }
    const rawText = splitThinking(raw, THINK).answer.trim()
    const json = rawText.replace(/^```(?:json)?\s*|\s*```$/g, "")
    try {
      return {
        data: JSON.parse(json) as unknown,
        rawText,
        usage: usageOf(next.value)
      }
    } catch {
      throw new Error(
        `The model's answer isn't JSON: ${rawText.slice(0, 200)}${rawText.length > 200 ? "…" : ""}`
      )
    }
  }
}

/**
 * A TanStack AI text adapter for a GGUF model this package lists
 * (`WLLAMA_TEXT_MODELS`), run on the CPU in the browser with wllama.
 *
 * @example
 * ```ts
 * chat({
 *   adapter: wllamaText("LiquidAI/LFM2.5-1.2B-Instruct-GGUF"),
 *   messages,
 *   threadId
 * })
 * ```
 */
export function wllamaText<TModel extends WllamaTextModel>(
  model: TModel,
  config: WllamaTextConfig = {}
): WllamaTextAdapter<TModel> {
  return new WllamaTextAdapter(config, model)
}
