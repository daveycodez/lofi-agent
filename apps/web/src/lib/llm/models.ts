import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import {
  APFEL_MODEL,
  APFEL_URL,
  apfelConnection
} from "#/lib/llm/apfel-connection"
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

export const modelInfo = (id: ModelId) =>
  MODELS.find((m) => m.id === id) ?? MODELS[0]

/** Whether Apple's on-device model is reachable through apfel. */
export const afmStatus = createStore<"checking" | "available" | "unavailable">(
  "checking"
)

let afmCheck: Promise<void> | undefined

/** Asks apfel's /v1/models whether the Apple model is there. Runs once. */
export function checkAfm() {
  afmCheck ??= fetch(APFEL_URL.replace("/chat/completions", "/models"), {
    signal: AbortSignal.timeout(2000)
  })
    .then((res) => (res.ok ? res.json() : null))
    .then((body: { data?: Array<{ id: string }> } | null) => {
      const found = body?.data?.some((m) => m.id === APFEL_MODEL) ?? false
      afmStatus.set(found ? "available" : "unavailable")
    })
    .catch(() => afmStatus.set("unavailable"))
  return afmCheck
}

/** The model the user picked, if any. */
export const modelStore = createStore<ModelId | null>(null, "lofi-agent:model")

/**
 * The model in use: the user's pick if it's available, otherwise Apple's
 * model when present, otherwise LFM3. Null while availability is unknown.
 */
function resolveModel(
  picked: ModelId | null,
  afm: ReturnType<typeof afmStatus.get>
): ModelId | null {
  if (afm === "checking") return null
  if (picked === "lfm3") return "lfm3"
  return afm === "available" ? "apple-fm" : "lfm3"
}

export const currentModel = () =>
  resolveModel(modelStore.get(), afmStatus.get())

export const useCurrentModel = () =>
  resolveModel(modelStore.use(), afmStatus.use())

/** Models to offer in the picker. */
export const useAvailableModels = () => {
  const afm = afmStatus.use()
  return MODELS.filter((m) => m.id !== "apple-fm" || afm === "available")
}

/** Sends each run to whichever model is selected when the user hits send. */
export const routedConnection: ConnectConnectionAdapter = {
  connect(messages, data, abortSignal, runContext) {
    return modelInfo(currentModel() ?? "lfm3").connection.connect(
      messages,
      data,
      abortSignal,
      runContext
    )
  }
}
