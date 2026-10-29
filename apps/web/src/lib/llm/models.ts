import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import { apfelConnection } from "#/lib/llm/apfel-connection"
import { lfm3Connection } from "#/lib/llm/lfm3-connection"
import { createStore } from "#/lib/store"

export type ModelId = "apple-fm" | "lfm3"

export const MODELS: Array<{
  id: ModelId
  name: string
  /** Provider logo in /public/icons (LobeHub icons, MIT). */
  icon: string
  connection: ConnectConnectionAdapter
}> = [
  {
    id: "apple-fm",
    name: "Apple Foundation Model",
    icon: "/icons/apple.svg",
    connection: apfelConnection
  },
  {
    id: "lfm3",
    name: "LFM3 1.7B",
    icon: "/icons/liquid.svg",
    connection: lfm3Connection
  }
]

export const modelStore = createStore<ModelId>("apple-fm", "lofi-agent:model")

export const modelInfo = (id: ModelId) =>
  MODELS.find((m) => m.id === id) ?? MODELS[0]

/** Sends each run to whichever model is selected when the user hits send. */
export const routedConnection: ConnectConnectionAdapter = {
  connect(messages, data, abortSignal, runContext) {
    return modelInfo(modelStore.get()).connection.connect(
      messages,
      data,
      abortSignal,
      runContext
    )
  }
}
