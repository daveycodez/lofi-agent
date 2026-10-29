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
  | { type: "done"; id: string }
  | { type: "error"; id: string; message: string }
