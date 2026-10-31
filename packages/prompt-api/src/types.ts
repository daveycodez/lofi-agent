/**
 * The parts of the Prompt API (https://github.com/webmachinelearning/prompt-api)
 * this package uses. TypeScript's DOM library has no types for it yet.
 * Shapes follow the spec and `@types/dom-chromium-ai`, kept here instead of
 * depending on that package: it declares `LanguageModel` as a global (in
 * every app that uses this package) and still types tools the way the spec
 * dropped (`execute` callbacks). Names follow the spec: `contextUsage`,
 * `contextWindow` and `contextoverflow` (web pages no longer have the old
 * `inputUsage`, `inputQuota` and `quotaoverflow`).
 */

export type Availability =
  "unavailable" | "downloadable" | "downloading" | "available"

/** Sampling presets, for web pages (the raw `temperature` and `topK` are for extensions). */
export type SamplingMode =
  | "most-predictable"
  | "predictable"
  | "slightly-predictable"
  | "balanced"
  | "slightly-creative"
  | "creative"
  | "most-creative"

export type LanguageModelMessageType =
  "text" | "image" | "audio" | "tool-call" | "tool-response"

export type LanguageModelExpected = {
  type: LanguageModelMessageType
  languages?: Array<string>
}

/** A tool the session may call (open loop: the page runs it). */
export type LanguageModelTool = {
  name: string
  description: string
  /** A JSON Schema with `type: "object"`. */
  inputSchema: object
}

/** A tool call, as the model writes it or as the page restores it. */
export type LanguageModelToolCall = {
  callId: string
  name: string
  arguments: Record<string, unknown>
}

/** One piece of a tool's result. */
export type LanguageModelToolResultContent = {
  type: "text" | "image" | "audio" | "object"
  value: unknown
}

/**
 * A piece of a message: text, an image (`Blob`, `ImageBitmap`, canvas...),
 * audio (`Blob`, `AudioBuffer`...), a tool call or a tool's response.
 */
export type LanguageModelContent = {
  type: LanguageModelMessageType
  value: unknown
}

export type LanguageModelMessage = {
  role: "system" | "user" | "assistant"
  content: string | Array<LanguageModelContent>
  /** On a last assistant message: the reply continues this text. */
  prefix?: boolean
}

/** Options `availability()` takes, which `create()` takes too. */
export type LanguageModelCoreOptions = {
  expectedInputs?: Array<LanguageModelExpected>
  expectedOutputs?: Array<LanguageModelExpected>
  tools?: Array<LanguageModelTool>
  /** Web pages get it only with Chrome's Sampling Parameters origin trial. */
  samplingMode?: SamplingMode
  /** Extensions only; web pages ignore them, and `samplingMode` wins. */
  temperature?: number
  topK?: number
}

export type LanguageModelCreateOptions = LanguageModelCoreOptions & {
  initialPrompts?: Array<LanguageModelMessage>
  /** Aborting it destroys the session, during `create()` or after. */
  signal?: AbortSignal
  monitor?: (monitor: EventTarget) => void
}

export type LanguageModelPromptOptions = {
  signal?: AbortSignal
  /** A JSON Schema (or RegExp) the reply must match. */
  responseConstraint?: object
  omitResponseConstraintInput?: boolean
}

export type LanguageModelInput = string | Array<LanguageModelMessage>

/**
 * What a prompt resolves to: a string, or, for a session that expects tool
 * calls, its pieces (text first, then tool calls).
 */
export type LanguageModelOutput = string | Array<LanguageModelContent>

export interface LanguageModelSession extends EventTarget {
  /** Tokens the session holds. */
  readonly contextUsage: number
  /** Tokens it can hold. */
  readonly contextWindow: number
  /** The preset the session samples with; null when made with raw parameters. */
  readonly samplingMode?: SamplingMode | null
  prompt(
    input: LanguageModelInput,
    options?: LanguageModelPromptOptions
  ): Promise<LanguageModelOutput>
  /**
   * Text as it's written; for a session that expects tool calls, pieces
   * (`{ type: "text", value }` deltas, then `{ type: "tool-call", value }`).
   */
  promptStreaming(
    input: LanguageModelInput,
    options?: LanguageModelPromptOptions
  ): ReadableStream<string | LanguageModelContent> &
    AsyncIterable<string | LanguageModelContent>
  append(
    input: LanguageModelInput,
    options?: { signal?: AbortSignal }
  ): Promise<void>
  measureContextUsage(
    input: LanguageModelInput,
    options?: LanguageModelPromptOptions
  ): Promise<number>
  clone(options?: { signal?: AbortSignal }): Promise<LanguageModelSession>
  destroy(): void
}

export type LanguageModelStatic = {
  availability(options?: LanguageModelCoreOptions): Promise<Availability>
  create(options?: LanguageModelCreateOptions): Promise<LanguageModelSession>
}

/** The `downloadprogress` event: `loaded` goes from 0 to 1. */
export type DownloadProgressEvent = Event & { loaded: number; total?: number }

/**
 * The `QuotaExceededError` a too-long input throws: `requested` tokens, and
 * the tokens free (`quota` in WebIDL's version of the error; Chrome's docs
 * call it `contextWindow`).
 */
export type QuotaExceededError = DOMException & {
  requested?: number
  quota?: number
  contextWindow?: number
}

/** The page's `LanguageModel`, or undefined where the browser has none. */
export const languageModel = () =>
  (globalThis as { LanguageModel?: LanguageModelStatic }).LanguageModel

type ToolCallClass = new (init: LanguageModelToolCall) => object
type ToolSuccessClass = new (init: {
  callId: string
  name: string
  result: Array<LanguageModelToolResultContent>
}) => object
type ToolErrorClass = new (init: {
  callId: string
  name: string
  errorMessage: string
}) => object

/**
 * The classes tool messages are made of, where the browser has native tool
 * use; undefined where it doesn't.
 */
export function toolClasses() {
  const g = globalThis as {
    LanguageModelToolCall?: ToolCallClass
    LanguageModelToolSuccess?: ToolSuccessClass
    LanguageModelToolError?: ToolErrorClass
  }
  if (
    !g.LanguageModelToolCall ||
    !g.LanguageModelToolSuccess ||
    !g.LanguageModelToolError
  ) {
    return undefined
  }
  return {
    ToolCall: g.LanguageModelToolCall,
    ToolSuccess: g.LanguageModelToolSuccess,
    ToolError: g.LanguageModelToolError
  }
}
