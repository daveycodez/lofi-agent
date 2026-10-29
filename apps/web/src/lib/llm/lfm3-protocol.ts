/** Shared between the page and the LFM3 worker. */
export const MODEL_ID = "onnx-community/LFM3-1.7B-ONNX"

export type ChatTurn = {
  role: "system" | "user" | "assistant"
  content: string
}

/** Experimental GPU upload strategies, picked with ?upload= (see lfm3.worker.ts). */
export type UploadMode = "default" | "write" | "write-wait"

/** Experiment: pause `ms` after every `everyMb` of weights read (0 = off). */
export type LoadPause = { ms: number; everyMb: number }

export type WorkerRequest =
  | { type: "load"; upload: UploadMode; pause: LoadPause }
  | { type: "generate"; id: string; chat: Array<ChatTurn> }
  | { type: "stop"; id: string }

export type WorkerResponse =
  | { type: "progress"; progress: number; cached: boolean }
  | { type: "ready" }
  | { type: "load-error"; message: string }
  | { type: "token"; id: string; text: string }
  | { type: "done"; id: string }
  | { type: "error"; id: string; message: string }
