/** Messages between the page and the local-model worker. */

export type ChatTurn = {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  /** The reply's thinking; LFM3's template keeps it for the current turn. */
  thinking?: string
  /** Past tool calls, for templates that write them themselves (LFM3). */
  tool_calls?: Array<{
    type: "function"
    function: { name: string; arguments: Record<string, unknown> }
  }>
}

/** A tool as chat templates take it (OpenAI's function format). */
export type ToolSpec = {
  type: "function"
  function: { name: string; description: string; parameters: unknown }
}

/** Sampling settings from each model's card. */
export type Sampling = {
  temperature: number
  top_k: number
  repetition_penalty: number
}

/** Token counts and timings of one generation, measured in the worker. */
export type RunTimings = {
  /** Tokens of the prompt, including any already in the model's cache. */
  inputTokens: number
  /** Tokens at the start of the prompt the model had read on an earlier run. */
  cachedTokens: number
  /** Tokens the model generated (the end-of-turn token included). */
  outputTokens: number
  /** From the request to the first generated token, in ms; null if none. */
  firstTokenMs: number | null
  /** From the first generated token to the last, in ms. */
  decodeMs: number
}

export type WorkerRequest =
  | { type: "load"; modelId: string }
  | {
      type: "generate"
      id: string
      chat: Array<ChatTurn>
      tools: Array<ToolSpec>
      thinking: boolean
      sampling: Sampling
    }
  | { type: "stop"; id: string }

export type WorkerResponse =
  | { type: "progress"; modelId: string; progress: number; cached: boolean }
  | { type: "ready"; modelId: string }
  | { type: "load-error"; modelId: string; message: string }
  | { type: "token"; id: string; text: string }
  /** The run's counts so far: once the prompt is tokenized, then as it replies. */
  | { type: "stats"; id: string; stats: RunTimings }
  | { type: "done"; id: string; stats: RunTimings }
  | { type: "error"; id: string; message: string }
  /** A diagnostic line for the page's console (dev only). */
  | { type: "log"; text: string }
