import type { ConnectConnectionAdapter } from "@tanstack/ai-react"
import { useSyncExternalStore } from "react"

import {
  APFEL_MODEL,
  APFEL_URL,
  apfelConnection
} from "#/lib/llm/apfel-connection"
import {
  loadLocalModel,
  localModelConnection,
  localModelStatus
} from "#/lib/llm/local-model"
import type { LocalModelStatus } from "#/lib/llm/local-model"
import { createStore } from "#/lib/store"

export type ModelId = "apple-fm" | "lfm3" | "lfm2.5"

/** Whether models that can think should reason before answering. */
export const thinkingStore = createStore<boolean>(false, "lofi-agent:thinking")

/** Models with a thinking mode (LFM2.5 Instruct has none). */
export const canThink = (id: ModelId) => id === "lfm3"

/** Models that run in the browser, by Hugging Face id. */
const LFM3_ID = "onnx-community/LFM3-1.7B-ONNX"
const LFM2_5_ID = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
const LOCAL_MODELS: Partial<Record<ModelId, string>> = {
  lfm3: LFM3_ID,
  "lfm2.5": LFM2_5_ID
}

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
    connection: localModelConnection(LFM3_ID, "LFM3", () => thinkingStore.get())
  },
  {
    id: "lfm2.5",
    name: "LFM2.5 1.2B",
    icon: "/icons/liquid.svg",
    connection: localModelConnection(LFM2_5_ID, "LFM2.5")
  }
]

/** Starts loading a local model (no-op for the Apple model). */
export function loadModel(id: ModelId) {
  const modelId = LOCAL_MODELS[id]
  if (modelId) void loadLocalModel(modelId).catch(() => undefined)
}

const IDLE = createStore<LocalModelStatus>({ state: "idle" })

/** Loading status of a model; the Apple model is always idle here. */
export const useModelStatus = (id: ModelId) => {
  const modelId = LOCAL_MODELS[id]
  return (modelId ? localModelStatus(modelId) : IDLE).use()
}

export const modelInfo = (id: ModelId) =>
  MODELS.find((m) => m.id === id) ?? MODELS[0]

/** The model the user picked. LFM3 runs anywhere, so it's the default. */
export const modelStore = createStore<ModelId>("lfm3", "lofi-agent:model")

/** Apple's model only exists on macOS and iOS (iPadOS reports itself as a Mac). */
const isApplePlatform = () =>
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad|iPod/.test(navigator.userAgent)

const noopSubscribe = () => () => undefined
const useIsApplePlatform = () =>
  useSyncExternalStore(noopSubscribe, isApplePlatform, () => false)

const onPlatform = (id: ModelId, isApple: boolean): ModelId =>
  id === "apple-fm" && !isApple ? "lfm3" : id

export const currentModel = () =>
  onPlatform(modelStore.get(), isApplePlatform())
export const useCurrentModel = () =>
  onPlatform(modelStore.use(), useIsApplePlatform())

/** Models to offer in the picker on this device. */
export const useModels = () => {
  const isApple = useIsApplePlatform()
  return MODELS.filter((m) => m.id !== "apple-fm" || isApple)
}

/** Result of the last Apple model check, shown next to the picker. */
export const afmStatus = createStore<"idle" | "checking" | "unavailable">(
  "idle"
)

/**
 * Asks apfel's /v1/models whether Apple's model is there. Reaching localhost
 * from a website makes Chrome ask for permission, so this only runs when the
 * user picks the Apple model.
 */
export async function checkAfm() {
  try {
    const res = await fetch(APFEL_URL.replace("/chat/completions", "/models"), {
      signal: AbortSignal.timeout(10_000)
    })
    const body = res.ok
      ? ((await res.json()) as { data?: Array<{ id: string }> })
      : null
    return body?.data?.some((m) => m.id === APFEL_MODEL) ?? false
  } catch {
    return false
  }
}

/** Sends each run to whichever model is selected when the user hits send. */
export const routedConnection: ConnectConnectionAdapter = {
  connect(messages, data, abortSignal, runContext) {
    return modelInfo(currentModel()).connection.connect(
      messages,
      data,
      abortSignal,
      runContext
    )
  }
}
