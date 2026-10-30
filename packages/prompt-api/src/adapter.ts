/**
 * A TanStack AI text adapter for the browser's built-in language model,
 * through the Prompt API (`LanguageModel`): Gemini Nano in Chrome,
 * Phi-4-mini in Edge. The browser downloads and runs the model itself;
 * nothing leaves the device. Hand it to `chat()` like any hosted model's
 * adapter: `chat()` runs tools and the agent loop, and each `chatStream`
 * call here is one step of the model, streamed back as AG-UI events.
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

import {
  baseOptions,
  DEFAULT_LANGUAGES,
  loadPromptApiModel,
  NO_PROMPT_API
} from "./model.js"
import { leaseSession } from "./sessions.js"
import type { Lease } from "./sessions.js"
import {
  decisionSchema,
  inputSchemaOf,
  parseArguments,
  parseDecision,
  toGuidedSchema,
  toolCallText,
  toolsPrompt,
  toPromptMessages
} from "./tools.js"
import { languageModel } from "./types.js"
import type {
  LanguageModelCreateOptions,
  LanguageModelMessage,
  LanguageModelSession
} from "./types.js"

/** The model: whichever one the browser provides. */
export type PromptApiTextModel = "built-in"

export type PromptApiTextConfig = {
  /**
   * Languages the conversation is in, input and output: ["en"] by default.
   * Chrome's model takes "en", "es", "ja", "de" and "fr", and checks its
   * safety filters for the output language.
   */
  languages?: ReadonlyArray<string>
}

/**
 * Per-call `modelOptions`. A session's sampling is set when it starts, so a
 * change starts a new session. Chrome ignores all of these on web pages
 * unless the page is in its Sampling Parameters origin trial.
 */
export type PromptApiModelOptions = {
  /** Sampling preset, "most-predictable" to "most-creative" ("balanced" by default). */
  samplingMode?:
    | "most-predictable"
    | "predictable"
    | "slightly-predictable"
    | "balanced"
    | "slightly-creative"
    | "creative"
    | "most-creative"
  /** Raw sampling, for extensions; give both or neither. */
  temperature?: number
  topK?: number
}

/**
 * Timings of one model call, in milliseconds, and the session's size, in
 * `RUN_FINISHED`'s `usage.providerUsageDetails`.
 */
export type PromptApiProviderUsageDetails = {
  /** From the call to the first generated text, deciding on tools included. */
  firstTokenMs?: number
  /** Reading the prompt: starting a session or appending to it, to the first text. */
  promptMs?: number
  /** From the first generated text to the last. */
  decodeMs?: number
  /** Tokens generated after the first one, over `decodeMs`. */
  decodeTokens?: number
  /** Tokens the session holds after the call. */
  contextUsage?: number
  /** Tokens it can hold. */
  contextWindow?: number
}

type PromptApiReasoning = never

type MetadataByModality = {
  text: unknown
  image: unknown
  audio: unknown
  video: unknown
  document: unknown
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

/** What a failed prompt means for the user. */
function promptErrorText(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "QuotaExceededError") {
      const { requested, quota } = error as DOMException & {
        requested?: number
        quota?: number
      }
      const sizes =
        requested && quota
          ? ` (it needs ${requested} tokens; the model holds ${quota})`
          : ""
      return `This is too long for the built-in model${sizes}. Start a new chat or send something shorter.`
    }
    if (error.name === "NotSupportedError") {
      return `The built-in model can't do this: ${error.message}`
    }
  }
  return `The built-in model failed: ${errorText(error)}`
}

/** A model call's token counts as TanStack AI usage, with its timings. */
function usageOf(
  promptTokens: number,
  completionTokens: number,
  cachedTokens: number,
  details: PromptApiProviderUsageDetails
): TokenUsage {
  const usage = buildBaseUsage<PromptApiProviderUsageDetails>({
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens
  })
  if (cachedTokens > 0) usage.promptTokensDetails = { cachedTokens }
  usage.providerUsageDetails = details
  return usage
}

/** Tokens a call wrote: the session's growth, else measured from its text. */
async function outputTokens(
  session: LanguageModelSession,
  grew: number,
  text: string
) {
  if (grew > 0) return grew
  // The session dropped old turns to make room: count the reply itself.
  return text ? session.measureContextUsage(text).catch(() => 0) : 0
}

/**
 * The tool calls of the current turn (since the last user message), as
 * the model writes them, to catch the model calling one again.
 */
function currentTurnCalls(messages: ReadonlyArray<ModelMessage>) {
  const start = messages.map((m) => m.role).lastIndexOf("user")
  return new Set(
    messages
      .slice(start + 1)
      .flatMap((m) => m.toolCalls ?? [])
      .map((call) =>
        toolCallText(
          call.function.name,
          parseArguments(call.function.arguments) ?? {}
        )
      )
  )
}

let ids = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(ids++).toString(36)}`

export class PromptApiTextAdapter extends BaseTextAdapter<
  PromptApiTextModel,
  PromptApiModelOptions,
  readonly ["text"],
  MetadataByModality,
  ReadonlyArray<string>,
  unknown,
  never,
  PromptApiReasoning
> {
  override readonly kind = "text" as const
  readonly name = "prompt-api" as const
  override readonly provider = "prompt-api" as const
  override readonly api = "prompt-api" as const
  override readonly inputModalities = ["text"] as const
  private readonly languages: ReadonlyArray<string>

  constructor(config: PromptApiTextConfig = {}) {
    super({}, "built-in")
    this.languages = config.languages ?? DEFAULT_LANGUAGES
  }

  /** Session options: text in the languages set, and any sampling asked for. */
  private createOptions(
    modelOptions: PromptApiModelOptions | undefined
  ): LanguageModelCreateOptions {
    const { samplingMode, temperature, topK } = modelOptions ?? {}
    return {
      ...baseOptions(this.languages),
      ...(samplingMode !== undefined && { samplingMode }),
      ...(temperature !== undefined && { temperature }),
      ...(topK !== undefined && { topK })
    }
  }

  async *chatStream(
    options: TextOptions<PromptApiModelOptions>
  ): AsyncIterable<AdapterYieldChunk> {
    const runId = options.runId ?? nextId("run")
    const threadId = options.threadId
    const messageId = nextId("msg")
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
    const startedAt = performance.now()

    yield {
      type: EventType.RUN_STARTED,
      runId,
      threadId,
      model,
      timestamp: timestamp(),
      ...(options.parentRunId && { parentRunId: options.parentRunId })
    }
    if (!languageModel()) {
      yield runError(NO_PROMPT_API)
      return
    }
    try {
      await loadPromptApiModel({ languages: this.languages, signal })
    } catch (error) {
      if (signal?.aborted) return
      yield runError(errorText(error))
      return
    }

    const tools = options.toolChoice === "none" ? [] : (options.tools ?? [])
    const system = [
      ...normalizeSystemPrompts(options.systemPrompts).map((p) => p.content),
      ...(tools.length > 0 ? [toolsPrompt(tools)] : [])
    ].join("\n\n")
    // What's new since the model's last message is this call's input; the
    // rest, a session may have read already.
    const conversation = toPromptMessages(options.messages)
    const lastReply = conversation.map((m) => m.role).lastIndexOf("assistant")
    const prior = conversation.slice(0, lastReply + 1)
    const input = conversation.slice(lastReply + 1)
    if (input.length === 0) {
      yield runError("There's no message for the built-in model to answer.")
      return
    }
    logger.request(
      `activity=chat provider=prompt-api model=${model} messages=${options.messages.length} tools=${tools.length} stream=true`,
      { provider: "prompt-api", model }
    )

    let lease: Lease
    try {
      lease = await leaseSession({
        threadId,
        system,
        prior,
        options: this.createOptions(options.modelOptions),
        signal
      })
    } catch (error) {
      if (signal?.aborted) return
      logger.errors("prompt-api session failed", {
        error,
        source: "prompt-api.chatStream"
      })
      yield runError(promptErrorText(error))
      return
    }
    const { session } = lease
    // Given back only once the call has ended well; freed otherwise (an
    // abort, an error, or the stream left unread).
    let released = false
    const contextWindow = session.contextWindow

    try {
      const before = session.contextUsage
      const inputTokens = await session.measureContextUsage(input)
      const call =
        tools.length > 0
          ? await this.decideToolCall(session, input, tools, options, signal)
          : undefined

      if (call) {
        const callText = toolCallText(call.name, call.args)
        // The session reads the step as the next request will send it.
        await session.append(
          [...input, { role: "assistant", content: callText }],
          { signal }
        )
        const decidedAt = performance.now()
        const completionTokens = await outputTokens(
          session,
          session.contextUsage - before - inputTokens,
          callText
        )
        lease.release([
          ...prior,
          ...input,
          { role: "assistant", content: callText }
        ])
        released = true
        const toolCallId = nextId("tool")
        const args = JSON.stringify(call.args)
        yield {
          type: EventType.TOOL_CALL_START,
          toolCallId,
          toolCallName: call.name,
          parentMessageId: messageId,
          index: 0,
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
        yield {
          type: EventType.RUN_FINISHED,
          runId,
          threadId,
          model,
          timestamp: timestamp(),
          finishReason: "tool_calls",
          usage: usageOf(
            before + inputTokens,
            completionTokens,
            lease.cachedTokens,
            {
              firstTokenMs: decidedAt - startedAt,
              contextUsage: session.contextUsage,
              contextWindow
            }
          )
        }
        return
      }

      // The answer, streamed as the model writes it.
      const promptAt = performance.now()
      let first = 0
      let last = 0
      let text = ""
      for await (const delta of session.promptStreaming(input, { signal })) {
        if (!delta) continue
        last = performance.now()
        if (!first) {
          first = last
          yield {
            type: EventType.TEXT_MESSAGE_START,
            messageId,
            role: "assistant",
            model,
            timestamp: timestamp()
          }
        }
        text += delta
        yield {
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId,
          delta,
          content: text,
          model,
          timestamp: timestamp()
        }
      }
      if (first) {
        yield {
          type: EventType.TEXT_MESSAGE_END,
          messageId,
          model,
          timestamp: timestamp()
        }
      }
      const completionTokens = await outputTokens(
        session,
        session.contextUsage - before - inputTokens,
        text
      )
      lease.release([...prior, ...input, { role: "assistant", content: text }])
      released = true
      yield {
        type: EventType.RUN_FINISHED,
        runId,
        threadId,
        model,
        timestamp: timestamp(),
        finishReason: "stop",
        usage: usageOf(
          before + inputTokens,
          completionTokens,
          lease.cachedTokens,
          {
            ...(first && {
              firstTokenMs: first - startedAt,
              promptMs: lease.createMs + (first - promptAt),
              decodeMs: last - first,
              decodeTokens: Math.max(completionTokens - 1, 0)
            }),
            contextUsage: session.contextUsage,
            contextWindow
          }
        )
      }
    } catch (error) {
      if (signal?.aborted) return
      logger.errors("prompt-api prompt failed", {
        error,
        source: "prompt-api.chatStream"
      })
      yield runError(promptErrorText(error))
    } finally {
      if (!released) lease.discard()
    }
  }

  /**
   * The tool the model calls for this input, with its arguments, or
   * undefined if it answers. Both are guided generations on copies of the
   * session, so the session itself reads only what the request will send.
   */
  private async decideToolCall(
    session: LanguageModelSession,
    input: Array<LanguageModelMessage>,
    tools: Array<Tool>,
    options: TextOptions<PromptApiModelOptions>,
    signal: AbortSignal | undefined
  ): Promise<{ name: string; args: Record<string, unknown> } | undefined> {
    const names = tools.map((tool) => tool.name)
    const choice = options.toolChoice
    let action: string | undefined
    if (typeof choice === "object" && names.includes(choice.name)) {
      action = choice.name
    } else {
      const raw = await promptCopy(session, input, signal, {
        responseConstraint: decisionSchema(names, choice === "required")
      })
      action = parseDecision(raw)?.action
      options.logger.provider("provider=prompt-api decision", { raw })
    }
    const tool = tools.find((t) => t.name === action)
    if (!tool) return undefined

    const schema = inputSchemaOf(tool)
    const hasFields =
      Object.keys((schema.properties ?? {}) as object).length > 0
    const raw = hasFields
      ? await promptCopy(session, input, signal, { responseConstraint: schema })
      : "{}"
    const args = parseArguments(raw) ?? {}
    // Small models sometimes call the tool they just called, with the same
    // arguments, instead of answering from its result.
    if (currentTurnCalls(options.messages).has(toolCallText(tool.name, args))) {
      return undefined
    }
    return { name: tool.name, args }
  }

  /**
   * A JSON answer for `chat({ outputSchema })`, from guided generation:
   * `responseConstraint` holds the model to the schema. A schema the
   * browser can't enforce is tried again in the subset it can. `chat()`
   * validates the result.
   */
  async structuredOutput(
    options: StructuredOutputOptions<PromptApiModelOptions>
  ): Promise<StructuredOutputResult<unknown>> {
    const { chatOptions, outputSchema } = options
    const signal = chatOptions.abortController?.signal
    await loadPromptApiModel({ languages: this.languages, signal })
    const conversation = toPromptMessages(chatOptions.messages)
    const lastReply = conversation.map((m) => m.role).lastIndexOf("assistant")
    const input = conversation.slice(lastReply + 1)
    const lease = await leaseSession({
      // A session of its own, freed after.
      threadId: undefined,
      system: normalizeSystemPrompts(chatOptions.systemPrompts)
        .map((p) => p.content)
        .join("\n\n"),
      prior: conversation.slice(0, lastReply + 1),
      options: this.createOptions(chatOptions.modelOptions),
      signal
    })
    const { session } = lease
    try {
      const before = session.contextUsage
      const inputTokens =
        input.length > 0 ? await session.measureContextUsage(input) : 0
      const ask = (schema: object) =>
        session.prompt(input.length > 0 ? input : "Answer.", {
          responseConstraint: schema,
          signal
        })
      let rawText: string
      try {
        rawText = await ask(outputSchema)
      } catch (error) {
        if (!(
          error instanceof DOMException && error.name === "NotSupportedError"
        ))
          throw error
        rawText = await ask(toGuidedSchema(outputSchema))
      }
      const completionTokens = await outputTokens(
        session,
        session.contextUsage - before - inputTokens,
        rawText
      )
      try {
        return {
          data: JSON.parse(rawText) as unknown,
          rawText,
          model: this.model,
          usage: usageOf(before + inputTokens, completionTokens, 0, {
            contextUsage: session.contextUsage,
            contextWindow: session.contextWindow
          })
        }
      } catch {
        throw new Error(
          `The model's answer isn't JSON: ${rawText.slice(0, 200)}${rawText.length > 200 ? "…" : ""}`
        )
      }
    } catch (error) {
      if (error instanceof DOMException) throw new Error(promptErrorText(error))
      throw error
    } finally {
      lease.discard()
    }
  }
}

/** A guided generation on a copy of the session, freed after. */
async function promptCopy(
  session: LanguageModelSession,
  input: Array<LanguageModelMessage>,
  signal: AbortSignal | undefined,
  options: { responseConstraint: object }
) {
  const copy = await session.clone({ signal })
  try {
    return await copy.prompt(input, { ...options, signal })
  } finally {
    copy.destroy()
  }
}

/**
 * A TanStack AI text adapter for the browser's built-in model, through the
 * Prompt API: Gemini Nano in Chrome, Phi-4-mini in Edge.
 *
 * @example
 * ```ts
 * if ((await promptApiAvailability()) !== "unavailable") {
 *   chat({ adapter: promptApiText(), messages, threadId })
 * }
 * ```
 */
export function promptApiText(
  config: PromptApiTextConfig = {}
): PromptApiTextAdapter {
  return new PromptApiTextAdapter(config)
}
