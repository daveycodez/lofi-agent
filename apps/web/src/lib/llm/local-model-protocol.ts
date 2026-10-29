/** Messages between the page and the local-model worker. */

export type ChatTurn = {
  role: "system" | "user" | "assistant"
  content: string
}

export type WorkerRequest =
  | { type: "load"; modelId: string }
  | { type: "generate"; id: string; chat: Array<ChatTurn> }
  | { type: "stop"; id: string }

export type WorkerResponse =
  | { type: "progress"; modelId: string; progress: number; cached: boolean }
  | { type: "ready"; modelId: string }
  | { type: "load-error"; modelId: string; message: string }
  | { type: "token"; id: string; text: string }
  | { type: "done"; id: string }
  | { type: "error"; id: string; message: string }
