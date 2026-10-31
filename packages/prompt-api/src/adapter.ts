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
  assistantMessage,
  mediaIn,
  toLanguageModelMessages,
  toPromptMessages
} from "./messages.js"
import type { PromptMessage } from "./messages.js"
import {
  baseOptions,
  DEFAULT_LANGUAGES,
  NO_PROMPT_API,
  PromptApiError
} from "./model.js"
import type { MediaInput } from "./model.js"
import { leaseSession } from "./sessions.js"
import type { Lease } from "./sessions.js"
import {
  decisionSchema,
  inputSchemaOf,
  noNativeTools,
  parseArguments,
  parseDecision,
  supportsNativeTools,
  toGuidedSchema,
  toLanguageModelTools,
  toolCallText,
  toolsPrompt
} from "./tools.js"
import { languageModel } from "./types.js"
import type {
  LanguageModelContent,
  LanguageModelCreateOptions,
  LanguageModelMessage,
  LanguageModelOutput,
  LanguageModelSession,
  LanguageModelToolCall,
  QuotaExceededError,
  SamplingMode
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
  /**
   * Whether to use the browser's own tool use where it has it (the spec's
   * open-loop tool use, which Chrome doesn't ship to web pages yet): true
   * by default. Off, or where the browser lacks it, tools go through guided
   * generation. A specific `toolChoice` always does.
   */
  nativeTools?: boolean
  /**
   * Called when a conversation outgrows the context window and the browser
   * drops its oldest turns (never the system prompt) to make room.
   */
  onContextOverflow?: (event: { threadId: string | undefined }) => void
}

/**
 * Per-call `modelOptions`. A session's sampling is set when it starts, so a
 * change starts a new session.
 */
export type PromptApiModelOptions = {
  /**
   * Sampling preset, "most-predictable" to "most-creative" ("balanced" by
   * default). Chrome takes it on web pages only in its Sampling Parameters
   * origin trial.
   */
  samplingMode?: SamplingMode
  /** Raw sampling, for extensions (web pages ignore it); `samplingMode` wins. */
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
  /**
   * Tokens generated after the first one, over `decodeMs`: an estimate (the
   * Prompt API counts tokens for a whole message, not as they're written).
   */
  decodeTokens?: number
  /** Tokens the session holds after the call. */
  contextUsage?: number
  /** Tokens it can hold. */
  contextWindow?: number
  /** True when the browser dropped the oldest turns to make room during the call. */
  contextOverflow?: boolean
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
  if (error instanceof PromptApiError) return error.message
  if (error instanceof DOMException) {
    if (error.name === "QuotaExceededError") {
      const { requested, quota, contextWindow } = error as QuotaExceededError
      const free = quota ?? contextWindow
      const sizes =
        requested && free !== undefined
          ? ` (it needs ${requested} tokens; ${free} are free)`
          : ""
      return `This is too long for the built-in model${sizes}. Start a new chat or send something shorter.`
    }
    if (error.name === "NotSupportedError") {
      return `The built-in model can't do this: ${error.message}`
    }
    if (error.name === "SecurityError") {
      return "The built-in model can't read that image: it's from another site that doesn't allow it (CORS)."
    }
    if (error.name === "EncodingError") {
      return "The built-in model can't read that image or audio: its format isn't supported."
    }
  }
  return `The built-in model failed: ${errorText(error)}`
}

/** Whether an error is the browser refusing a session with tools. */
function refusedTools(error: unknown) {
  const cause = error instanceof PromptApiError ? error.cause : error
  return (
    cause instanceof TypeError ||
    (cause instanceof DOMException && cause.name === "NotSupportedError")
  )
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

/**
 * Tokens of what the model wrote, measured once it's done (so the count
 * never holds up the reply). The rest of the session is the prompt.
 */
const replyTokens = (session: LanguageModelSession, text: string) =>
  text ? session.measureContextUsage(text).catch(() => 0) : Promise.resolve(0)

/** A prompt's reply as text, from a session that may answer in pieces. */
const textOf = (output: LanguageModelOutput) =>
  typeof output === "string"
    ? output
    : output
        .map((part) => (part.type === "text" ? String(part.value) : ""))
        .join("")

/** A streamed chunk's pieces: a text delta, or pieces with tool calls. */
const piecesOf = (
  chunk: string | LanguageModelContent | Array<LanguageModelContent>
): Array<LanguageModelContent> =>
  typeof chunk === "string"
    ? [{ type: "text", value: chunk }]
    : Array.isArray(chunk)
      ? chunk
      : [chunk]

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

/** The conversation split where the model last spoke. */
function splitAtLastReply(conversation: Array<PromptMessage>) {
  const lastReply = conversation.map((m) => m.role).lastIndexOf("assistant")
  return {
    prior: conversation.slice(0, lastReply + 1),
    input: conversation.slice(lastReply + 1)
  }
}

let ids = 0
const nextId = (prefix: string) =>
  `${prefix}_${Date.now().toString(36)}_${(ids++).toString(36)}`

/** A call's session and what it reads. */
type Prepared = {
  lease: Lease
  prior: Array<PromptMessage>
  inputMessages: Array<PromptMessage>
  input: Array<LanguageModelMessage>
}

export class PromptApiTextAdapter extends BaseTextAdapter<
  PromptApiTextModel,
  PromptApiModelOptions,
  readonly ["text", "image", "audio"],
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
  override readonly inputModalities = ["text", "image", "audio"] as const
  private readonly languages: ReadonlyArray<string>
  private readonly nativeTools: boolean
  private readonly onContextOverflow: PromptApiTextConfig["onContextOverflow"]

  constructor(config: PromptApiTextConfig = {}) {
    super({}, "built-in")
    this.languages = config.languages ?? DEFAULT_LANGUAGES
    this.nativeTools = config.nativeTools ?? true
    this.onContextOverflow = config.onContextOverflow
  }

  /**
   * Session options: text in the languages set, the conversation's images
   * and audio, the browser's own tools if it runs them, and any sampling.
   */
  private createOptions(
    modelOptions: PromptApiModelOptions | undefined,
    {
      media,
      tools
    }: { media: ReadonlyArray<MediaInput>; tools?: ReadonlyArray<Tool> }
  ): LanguageModelCreateOptions {
    const { samplingMode, temperature, topK } = modelOptions ?? {}
    return {
      ...baseOptions(this.languages, { media, nativeTools: !!tools }),
      ...(tools && { tools: toLanguageModelTools(tools) }),
      ...(samplingMode !== undefined && { samplingMode }),
      ...(temperature !== undefined && { temperature }),
      ...(topK !== undefined && { topK })
    }
  }

  /** The session for a call, having read everything before its input. */
  private async prepare(
    options: TextOptions<PromptApiModelOptions>,
    tools: Array<Tool>,
    native: boolean,
    signal: AbortSignal | undefined
  ): Promise<Prepared> {
    const conversation = toPromptMessages(options.messages, {
      nativeTools: native
    })
    // What's new since the model's last message is this call's input; the
    // rest, a session may have read already.
    const { prior, input: inputMessages } = splitAtLastReply(conversation)
    if (inputMessages.length === 0) {
      throw new PromptApiError(
        "There's no message for the built-in model to answer."
      )
    }
    const system = [
      ...normalizeSystemPrompts(options.systemPrompts).map((p) => p.content),
      ...(tools.length > 0 && !native ? [toolsPrompt(tools)] : [])
    ].join("\n\n")
    const media = new Map<string, Promise<Blob>>()
    const lease = await leaseSession({
      threadId: options.threadId,
      system,
      prior,
      options: this.createOptions(options.modelOptions, {
        media: mediaIn(conversation),
        tools: native ? tools : undefined
      }),
      signal,
      media
    })
    if (lease.miss) {
      options.logger.provider(
        `provider=prompt-api new session: ${lease.miss}`,
        { threadId: options.threadId }
      )
    }
    try {
      const input = await toLanguageModelMessages(inputMessages, media)
      return { lease, prior, inputMessages, input }
    } catch (error) {
      lease.discard()
      throw error
    }
  }

  /** The session's size, and whether it dropped old turns, for usage. */
  private sessionDetails(
    lease: Lease,
    threadId: string | undefined,
    logger: TextOptions["logger"]
  ): PromptApiProviderUsageDetails {
    const overflowed = lease.overflowed()
    if (overflowed) {
      logger.provider(
        "provider=prompt-api context overflow: the oldest turns were dropped",
        { threadId }
      )
      this.onContextOverflow?.({ threadId })
    }
    return {
      contextUsage: lease.session.contextUsage,
      contextWindow: lease.session.contextWindow,
      ...(overflowed && { contextOverflow: true })
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
    const toolCallEvents = (
      toolCallId: string,
      name: string,
      input: Record<string, unknown>,
      index: number
    ): Array<AdapterYieldChunk> => {
      const args = JSON.stringify(input)
      return [
        {
          type: EventType.TOOL_CALL_START,
          toolCallId,
          toolCallName: name,
          parentMessageId: messageId,
          index,
          model,
          timestamp: timestamp()
        },
        {
          type: EventType.TOOL_CALL_ARGS,
          toolCallId,
          delta: args,
          args,
          model,
          timestamp: timestamp()
        },
        {
          type: EventType.TOOL_CALL_END,
          toolCallId,
          toolCallName: name,
          args,
          input,
          model,
          timestamp: timestamp()
        }
      ]
    }
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

    const tools = options.toolChoice === "none" ? [] : (options.tools ?? [])
    const choice = options.toolChoice
    let native =
      tools.length > 0 &&
      this.nativeTools &&
      (choice === undefined || choice === "auto") &&
      (await supportsNativeTools(this.languages))
    logger.request(
      `activity=chat provider=prompt-api model=${model} messages=${options.messages.length} tools=${tools.length} nativeTools=${native} stream=true`,
      { provider: "prompt-api", model }
    )

    // A kept session the browser destroyed (it shows only once used) is
    // replaced once, if nothing was sent yet.
    let retried = false
    for (;;) {
      let prepared: Prepared
      for (;;) {
        try {
          prepared = await this.prepare(options, tools, native, signal)
          break
        } catch (error) {
          if (signal?.aborted) return
          if (native && refusedTools(error)) {
            // The browser said it runs tools, then refused a session with them.
            logger.provider("provider=prompt-api native tools refused", {
              error
            })
            noNativeTools(this.languages)
            native = false
            continue
          }
          logger.errors("prompt-api session failed", {
            error,
            source: "prompt-api.chatStream"
          })
          yield runError(promptErrorText(error))
          return
        }
      }
      const { lease, prior, inputMessages, input } = prepared
      const { session } = lease
      // Given back only once the call has ended well; freed otherwise (an
      // abort, an error, or the stream left unread).
      let released = false
      let sent = false

      try {
        const call =
          tools.length > 0 && !native
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
          const toolCallId = nextId("tool")
          const completionTokens = await replyTokens(session, callText)
          lease.release([
            ...prior,
            ...inputMessages,
            { role: "assistant", content: callText }
          ])
          released = true
          yield* toolCallEvents(toolCallId, call.name, call.args, 0)
          yield {
            type: EventType.RUN_FINISHED,
            runId,
            threadId,
            model,
            timestamp: timestamp(),
            finishReason: "tool_calls",
            usage: usageOf(
              session.contextUsage - completionTokens,
              completionTokens,
              lease.cachedTokens,
              {
                firstTokenMs: decidedAt - startedAt,
                ...this.sessionDetails(lease, threadId, logger)
              }
            )
          }
          return
        }

        // The answer, streamed as the model writes it; with the browser's
        // own tools, its tool calls come after any text.
        const promptAt = performance.now()
        let first = 0
        let last = 0
        let text = ""
        const written: Array<LanguageModelToolCall> = []
        for await (const chunk of session.promptStreaming(input, { signal })) {
          for (const piece of piecesOf(chunk)) {
            if (piece.type === "tool-call") {
              written.push(piece.value as LanguageModelToolCall)
              continue
            }
            if (piece.type !== "text" || typeof piece.value !== "string")
              continue
            const delta = piece.value
            if (!delta) continue
            last = performance.now()
            if (!first) {
              first = last
              sent = true
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
        }
        if (first) {
          yield {
            type: EventType.TEXT_MESSAGE_END,
            messageId,
            model,
            timestamp: timestamp()
          }
        }
        const calls = written.map((c) => ({
          callId: c.callId || nextId("tool"),
          name: c.name,
          arguments: c.arguments ?? {}
        }))
        const completionTokens = await replyTokens(
          session,
          [text, ...calls.map((c) => toolCallText(c.name, c.arguments))]
            .filter(Boolean)
            .join("\n")
        )
        const reply = assistantMessage(text, calls, native)
        lease.release([...prior, ...inputMessages, ...(reply ? [reply] : [])])
        released = true
        for (const [index, c] of calls.entries()) {
          yield* toolCallEvents(c.callId, c.name, c.arguments, index)
        }
        yield {
          type: EventType.RUN_FINISHED,
          runId,
          threadId,
          model,
          timestamp: timestamp(),
          finishReason: calls.length > 0 ? "tool_calls" : "stop",
          usage: usageOf(
            session.contextUsage - completionTokens,
            completionTokens,
            lease.cachedTokens,
            {
              ...(first && {
                firstTokenMs: first - startedAt,
                promptMs: lease.createMs + (first - promptAt),
                decodeMs: last - first,
                decodeTokens: Math.max(completionTokens - 1, 0)
              }),
              ...this.sessionDetails(lease, threadId, logger)
            }
          )
        }
        return
      } catch (error) {
        if (signal?.aborted) return
        const destroyed =
          error instanceof DOMException && error.name === "InvalidStateError"
        if (destroyed && lease.reused && !sent && !retried) {
          logger.provider(
            "provider=prompt-api the kept session was destroyed; starting a new one",
            { threadId }
          )
          retried = true
          continue
        }
        logger.errors("prompt-api prompt failed", {
          error,
          source: "prompt-api.chatStream"
        })
        yield runError(promptErrorText(error))
        return
      } finally {
        if (!released) lease.discard()
      }
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
    const conversation = toPromptMessages(chatOptions.messages, {
      nativeTools: false
    })
    const { prior, input: inputMessages } = splitAtLastReply(conversation)
    const media = new Map<string, Promise<Blob>>()
    const lease = await leaseSession({
      // A session of its own, freed after.
      threadId: undefined,
      system: normalizeSystemPrompts(chatOptions.systemPrompts)
        .map((p) => p.content)
        .join("\n\n"),
      prior,
      options: this.createOptions(chatOptions.modelOptions, {
        media: mediaIn(conversation)
      }),
      signal,
      media
    })
    const { session } = lease
    try {
      const input =
        inputMessages.length > 0
          ? await toLanguageModelMessages(inputMessages, media)
          : "Answer."
      const ask = async (schema: object) =>
        textOf(
          await session.prompt(input, { responseConstraint: schema, signal })
        )
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
      const completionTokens = await replyTokens(session, rawText)
      try {
        return {
          data: JSON.parse(rawText) as unknown,
          rawText,
          model: this.model,
          usage: usageOf(
            session.contextUsage - completionTokens,
            completionTokens,
            0,
            {
              contextUsage: session.contextUsage,
              contextWindow: session.contextWindow
            }
          )
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
    return textOf(await copy.prompt(input, { ...options, signal }))
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
