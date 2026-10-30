/**
 * A TanStack AI text adapter for GGUF models that run in the browser on the
 * CPU, with wllama (llama.cpp in WebAssembly). Hand it to `chat()` like any
 * hosted model's adapter: `chat()` runs tools and the agent loop, and each
 * `chatStream` call here is one chat completion from llama.cpp's
 * OpenAI-compatible API, streamed back as AG-UI events. llama.cpp writes
 * the prompt with the model's chat template and reads its thinking and
 * tool calls back.
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
import type {
  ChatCompletionMessage,
  ChatCompletionTool,
  ResultTimings
} from "@wllama/wllama/esm/index.js"

import {
  completeChat,
  configureRuntime,
  contextError,
  streamChat
} from "./client.js"
import type { ChatRequest } from "./client.js"
import { modelMeta } from "./model-meta.js"
import type {
  Sampling,
  WllamaModelReasoningByName,
  WllamaTextModel
} from "./model-meta.js"

export type WllamaTextConfig = {
  /**
   * Where wllama's WebAssembly file is served, for bundlers that don't
   * emit it from `new URL(…, import.meta.url)`. Applies to every model.
   */
  wasmUrl?: string
  /**
   * Where this package's faster build (`wasm/wllama-relaxed.wasm`, for
   * browsers with relaxed SIMD) is served, if not emitted by the bundler, or
   * `false` to use wllama's own build everywhere.
   */
  relaxedSimdWasmUrl?: string | false
  /**
   * Where wllama's compat build (for Safari) is served, if not from
   * `@wllama/wllama-compat` through the bundler.
   */
  compat?: { wasm: string; worker: string }
  /** CPU threads. wllama's default otherwise: half the cores. */
  threads?: number
  /** wllama's and llama.cpp's logs, in the console. */
  debug?: boolean
}

/** Per-call `modelOptions`: sampling over the model card's, and the most tokens to generate. */
export type WllamaTextProviderOptions = Partial<Sampling> & {
  /** Most tokens to generate. llama.cpp's default otherwise: up to the context limit. */
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

/** The conversation as OpenAI-style chat messages, the system prompt first. */
function toMessages(
  systemPrompts: TextOptions["systemPrompts"],
  messages: Array<ModelMessage>,
  extraSystem?: string
): Array<ChatCompletionMessage> {
  const system = normalizeSystemPrompts(systemPrompts)
    .map((p) => p.content)
    .concat(extraSystem ?? [])
    .join("\n\n")
  const out: Array<ChatCompletionMessage> = system
    ? [{ role: "system", content: system }]
    : []
  for (const m of messages) {
    if (m.role === "tool") {
      out.push({
        role: "tool",
        tool_call_id: m.toolCallId ?? "",
        content: toolResultText(m)
      })
    } else if (m.role === "assistant") {
      const thinking = m.thinking?.map((t) => t.content).join("")
      out.push({
        role: "assistant",
        content: contentText(m.content),
        ...(m.toolCalls?.length && {
          tool_calls: m.toolCalls.map((c) => ({
            id: c.id,
            type: "function" as const,
            function: { name: c.function.name, arguments: c.function.arguments }
          }))
        }),
        // llama.cpp hands it to the chat template, which keeps or drops it.
        ...(thinking && { reasoning_content: thinking })
      })
    } else {
      out.push({ role: m.role, content: contentText(m.content) })
    }
  }
  return out
}

/** Tools in OpenAI's function format. */
function toTools(tools: Array<Tool> | undefined): Array<ChatCompletionTool> {
  return (tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: (tool.inputSchema ?? {
        type: "object",
        properties: {}
      }) as ChatCompletionTool["function"]["parameters"]
    }
  }))
}

/** A model call's counts as TanStack AI usage, its timings as provider details. */
function usageOf(
  timings: ResultTimings | undefined,
  firstTokenMs: number | null
): TokenUsage | undefined {
  if (!timings) return undefined
  const promptTokens = timings.prompt_n + timings.cache_n
  const usage = buildBaseUsage<WllamaProviderUsageDetails>({
    promptTokens,
    completionTokens: timings.predicted_n,
    totalTokens: promptTokens + timings.predicted_n
  })
  if (timings.cache_n > 0) {
    usage.promptTokensDetails = { cachedTokens: timings.cache_n }
  }
  usage.providerUsageDetails = {
    ...(firstTokenMs !== null && { firstTokenMs }),
    promptMs: timings.prompt_ms,
    ...(timings.predicted_n > 0 && {
      decodeMs: timings.predicted_ms,
      decodeTokens: timings.predicted_n
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
      ...(config.relaxedSimdWasmUrl !== undefined && {
        relaxedSimdWasmUrl: config.relaxedSimdWasmUrl
      }),
      ...(config.compat && { compat: config.compat }),
      ...(config.threads && { threads: config.threads }),
      ...(config.debug && { debug: true })
    })
  }

  /** Sampling and length for a call: `modelOptions` over the model card's. */
  private settings(modelOptions: WllamaTextProviderOptions | undefined) {
    const { max_new_tokens, ...sampling } = modelOptions ?? {}
    return {
      ...modelMeta(this.model).sampling,
      ...sampling,
      ...(max_new_tokens !== undefined && { max_tokens: max_new_tokens })
    } as Partial<ChatRequest>
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
    const { context } = modelMeta(model)
    const timestamp = () => Date.now()

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }

    let reasoning: "none" | "open" | "closed" = "none"
    let text = ""
    let textStarted = false
    /** Tool calls by their index in the stream. */
    const toolCalls = new Map<
      number,
      { id: string; name: string; args: string; started: boolean }
    >()
    let finishReason: string | null = null
    let timings: ResultTimings | undefined
    let firstTokenMs: number | null = null
    let failure: string | undefined
    const requested = performance.now()

    const endReasoning = function* () {
      if (reasoning !== "open") return
      reasoning = "closed"
      yield {
        type: EventType.REASONING_MESSAGE_END,
        messageId: reasoningId,
        model,
        timestamp: timestamp()
      } satisfies AdapterYieldChunk
      yield {
        type: EventType.REASONING_END,
        messageId: reasoningId,
        model,
        timestamp: timestamp()
      } satisfies AdapterYieldChunk
    }

    try {
      const stream = await streamChat(
        model,
        {
          messages: toMessages(options.systemPrompts, options.messages),
          ...(options.tools?.length && { tools: toTools(options.tools) }),
          ...this.settings(options.modelOptions)
        },
        signal
      ).catch((error: unknown) => {
        throw new Error(`Couldn't load ${model}: ${errorText(error)}`)
      })
      for await (const chunk of stream) {
        timings = chunk.timings ?? timings
        const choice = chunk.choices?.[0]
        if (!choice) continue
        finishReason = choice.finish_reason ?? finishReason
        const delta = choice.delta as typeof choice.delta & {
          reasoning_content?: string | null
        }
        if (
          firstTokenMs === null &&
          (delta.reasoning_content || delta.content || delta.tool_calls)
        ) {
          firstTokenMs = performance.now() - requested
        }

        if (delta.reasoning_content) {
          if (reasoning === "none") {
            reasoning = "open"
            yield {
              type: EventType.REASONING_START,
              messageId: reasoningId,
              model,
              timestamp: timestamp()
            }
            yield {
              type: EventType.REASONING_MESSAGE_START,
              messageId: reasoningId,
              role: "reasoning",
              model,
              timestamp: timestamp()
            }
          }
          yield {
            type: EventType.REASONING_MESSAGE_CONTENT,
            messageId: reasoningId,
            delta: delta.reasoning_content,
            model,
            timestamp: timestamp()
          }
        }

        // The answer starts at its first visible character: the newlines
        // after thinking, or before a tool call, aren't a reply.
        const piece = textStarted
          ? (delta.content ?? "")
          : (delta.content ?? "").trimStart()
        if (piece) {
          yield* endReasoning()
          if (!textStarted) {
            textStarted = true
            yield {
              type: EventType.TEXT_MESSAGE_START,
              messageId,
              role: "assistant",
              model,
              timestamp: timestamp()
            }
          }
          text += piece
          yield {
            type: EventType.TEXT_MESSAGE_CONTENT,
            messageId,
            delta: piece,
            content: text,
            model,
            timestamp: timestamp()
          }
        }

        for (const part of delta.tool_calls ?? []) {
          yield* endReasoning()
          let call = toolCalls.get(part.index)
          if (!call) {
            call = { id: part.id ?? nextId("tool"), name: "", args: "", started: false }
            toolCalls.set(part.index, call)
          }
          if (part.function?.name) call.name += part.function.name
          if (!call.started && call.name) {
            call.started = true
            yield {
              type: EventType.TOOL_CALL_START,
              toolCallId: call.id,
              toolCallName: call.name,
              parentMessageId: messageId,
              index: part.index,
              model,
              timestamp: timestamp()
            }
          }
          const args = part.function?.arguments ?? ""
          if (args) {
            call.args += args
            yield {
              type: EventType.TOOL_CALL_ARGS,
              toolCallId: call.id,
              delta: args,
              args: call.args,
              model,
              timestamp: timestamp()
            }
          }
        }
      }
    } catch (error) {
      // Stopped: the reply so far stands.
      if (!signal?.aborted) failure = errorText(contextError(error, context))
    }

    yield* endReasoning()
    for (const call of toolCalls.values()) {
      if (!call.started) continue
      let input: unknown = {}
      try {
        input = call.args ? JSON.parse(call.args) : {}
      } catch {
        // `chat()` reports arguments that aren't JSON to the model.
      }
      yield {
        type: EventType.TOOL_CALL_END,
        toolCallId: call.id,
        toolCallName: call.name,
        args: call.args,
        input,
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

    // Cut off by the context limit, not by a length the caller asked for.
    if (
      !failure &&
      finishReason === "length" &&
      timings &&
      timings.prompt_n + timings.cache_n + timings.predicted_n >= context
    ) {
      failure = `The reply stopped at the model's limit of ${new Intl.NumberFormat("en-US").format(context)} tokens for a session. Start a new session to keep going.`
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
    const usage = usageOf(timings, firstTokenMs)
    yield {
      type: EventType.RUN_FINISHED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      finishReason:
        toolCalls.size > 0
          ? "tool_calls"
          : finishReason === "length"
            ? "length"
            : "stop",
      ...(usage && { usage })
    }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`, constrained to the schema
   * by llama.cpp's JSON Schema grammar; `chat()` validates the result.
   */
  async structuredOutput(
    options: StructuredOutputOptions<WllamaTextProviderOptions>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const response = await completeChat(
      this.model,
      {
        messages: toMessages(chatOptions.systemPrompts, chatOptions.messages),
        response_format: {
          type: "json_schema",
          json_schema: { name: "output", schema: outputSchema, strict: true }
        },
        ...this.settings(chatOptions.modelOptions)
      },
      chatOptions.abortController?.signal
    )
    const rawText = response.choices[0]?.message.content?.trim() ?? ""
    const timings = (response as { timings?: ResultTimings }).timings
    try {
      return {
        data: JSON.parse(rawText) as unknown,
        rawText,
        ...(usageOf(timings, null) && { usage: usageOf(timings, null) })
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
