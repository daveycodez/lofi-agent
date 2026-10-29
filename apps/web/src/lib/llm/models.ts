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

export type ModelId = "apple-fm" | "lfm3" | "lfm3-web" | "lfm2.5"

/** Whether models that can think should reason before answering. */
export const thinkingStore = createStore<boolean>(false, "lofi-agent:thinking")

/** Models that can think before answering. */
export const canThink = (id: ModelId) =>
  id === "lfm3" || id === "lfm3-web" || id === "lfm2.5"

/** Models that run in the browser, by Hugging Face id. */
const LFM3_ID = "onnx-community/LFM3-1.7B-ONNX"
/**
 * A smaller build of LFM3 (GPTQ 4-bit, 905 MB of weights instead of 1.09 GB,
 * without the embedding copy onnxruntime makes for the original graph) made
 * to fit iPhone Safari's memory limit. Sequences are limited to 16,384
 * tokens. Pinned to a commit, so browsers fetch this version's graph.
 */
const LFM3_WEB_ID =
  "daveycodez/LFM3-1.7B-Web@a405c7fced5b66782eb5eea9fdb2fb3d5e2f8124"
const LFM2_5_ID = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
/** LFM2.5 has no thinking switch: thinking is a separate model. */
const LFM2_5_THINKING_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX"
const LOCAL_MODELS: Partial<Record<ModelId, string>> = {
  lfm3: LFM3_ID,
  "lfm3-web": LFM3_WEB_ID,
  "lfm2.5": LFM2_5_ID
}

/** LFM3 card, "General" (chat, tool use, agents, coding). */
const lfm3Connection = (modelId: string, name: string) =>
  localModelConnection({
    modelId,
    name,
    sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.1 },
    toolFormat: "lfm3",
    thinking: () => thinkingStore.get()
  })

// LFM2.5-1.2B-Instruct card.
const lfm25 = localModelConnection({
  modelId: LFM2_5_ID,
  name: "LFM2.5",
  sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.05 },
  toolFormat: "lfm2"
})
// LFM2.5-1.2B-Thinking card.
const lfm25Thinking = localModelConnection({
  modelId: LFM2_5_THINKING_ID,
  name: "LFM2.5 Thinking",
  sampling: { temperature: 0.05, top_k: 50, repetition_penalty: 1.05 },
  toolFormat: "lfm2",
  // Always thinks: give it the longer thinking budget.
  thinking: () => true
})

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
    connection: lfm3Connection(LFM3_ID, "LFM3")
  },
  {
    id: "lfm3-web",
    name: "LFM3 1.7B Web",
    icon: "/icons/liquid.svg",
    connection: lfm3Connection(LFM3_WEB_ID, "LFM3 Web")
  },
  {
    id: "lfm2.5",
    name: "LFM2.5 1.2B",
    icon: "/icons/liquid.svg",
    // The Think toggle picks the Thinking model when the message is sent.
    connection: {
      connect: (...args) =>
        (thinkingStore.get() ? lfm25Thinking : lfm25).connect(...args)
    }
  }
]

/** The Hugging Face model to run for `id`, given the Think toggle. */
const localModelFor = (id: ModelId, thinking: boolean) =>
  id === "lfm2.5" && thinking ? LFM2_5_THINKING_ID : LOCAL_MODELS[id]

/** Starts loading a local model (no-op for the Apple model). */
export function loadModel(id: ModelId) {
  const modelId = localModelFor(id, thinkingStore.get())
  if (modelId) void loadLocalModel(modelId).catch(() => undefined)
}

const IDLE = createStore<LocalModelStatus>({ state: "idle" })

/** Loading status of a model; the Apple model is always idle here. */
export const useModelStatus = (id: ModelId) => {
  const modelId = localModelFor(id, thinkingStore.use())
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
