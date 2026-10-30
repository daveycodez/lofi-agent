/**
 * The parts of the Prompt API (https://github.com/webmachinelearning/prompt-api)
 * this package uses, as Chrome ships them to web pages. TypeScript's DOM
 * library has no types for it yet. Names follow the spec: `contextUsage` and
 * `contextWindow` (formerly `inputUsage` and `inputQuota`).
 */

export type Availability =
  "unavailable" | "downloadable" | "downloading" | "available"

export type LanguageModelExpected = {
  type: "text" | "image" | "audio"
  languages?: Array<string>
}

export type LanguageModelMessage = {
  role: "system" | "user" | "assistant"
  content: string
  /** On a last assistant message: the reply continues this text. */
  prefix?: boolean
}

export type LanguageModelCreateOptions = {
  initialPrompts?: Array<LanguageModelMessage>
  expectedInputs?: Array<LanguageModelExpected>
  expectedOutputs?: Array<LanguageModelExpected>
  signal?: AbortSignal
  monitor?: (monitor: EventTarget) => void
  /** Sampling preset; web pages get it only with Chrome's origin trial. */
  samplingMode?: string
  /** Extensions only; web pages ignore them. */
  temperature?: number
  topK?: number
}

export type LanguageModelPromptOptions = {
  signal?: AbortSignal
  /** A JSON Schema (or RegExp) the reply must match. */
  responseConstraint?: object
  omitResponseConstraintInput?: boolean
}

export type LanguageModelInput = string | Array<LanguageModelMessage>

export interface LanguageModelSession extends EventTarget {
  /** Tokens the session holds. */
  readonly contextUsage: number
  /** Tokens it can hold. */
  readonly contextWindow: number
  prompt(
    input: LanguageModelInput,
    options?: LanguageModelPromptOptions
  ): Promise<string>
  promptStreaming(
    input: LanguageModelInput,
    options?: LanguageModelPromptOptions
  ): ReadableStream<string> & AsyncIterable<string>
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
  availability(options?: LanguageModelCreateOptions): Promise<Availability>
  create(options?: LanguageModelCreateOptions): Promise<LanguageModelSession>
}

/** The `downloadprogress` event: `loaded` goes from 0 to 1. */
export type DownloadProgressEvent = Event & { loaded: number; total?: number }

/** The page's `LanguageModel`, or undefined where the browser has none. */
export const languageModel = () =>
  (globalThis as { LanguageModel?: LanguageModelStatic }).LanguageModel
