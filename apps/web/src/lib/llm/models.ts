import type { ConnectConnectionAdapter } from "@tanstack/ai-react"
import { useSyncExternalStore } from "react"

import {
  APFEL_MODEL,
  APFEL_URL,
  apfelConnection
} from "#/lib/llm/apfel-connection"
import { localModelConnection, localModelStatus } from "#/lib/llm/local-model"
import type { LocalModelStatus } from "#/lib/llm/local-model"
import { createStore } from "#/lib/store"

export type ModelId = "apple-fm" | "lfm3" | "lfm3-web" | "lfm2.5" | "lfm2.5-web"

/**
 * Whether models that can think should reason before answering. On by
 * default: LFM3 and LFM2.5 answer much better with it. (A new key, so the
 * old default of off doesn't carry over.)
 */
export const thinkingStore = createStore<boolean>(true, "lofi-agent:think")

/** Models that can think before answering. */
export const canThink = (id: ModelId) =>
  id === "lfm3" || id === "lfm3-web" || id === "lfm2.5" || id === "lfm2.5-web"

/** Models that run in the browser, by Hugging Face id. */
const LFM3_ID = "onnx-community/LFM3-1.7B-ONNX"
/**
 * A build of LFM3 made for iPhone Safari: GPTQ 4-bit with block size 32 (so
 * onnxruntime-web's fast prompt kernel and fused kernels apply), a 48k
 * vocabulary, no embedding copy at load (900 MB of weights instead of
 * 1.09 GB), and flash attention in the sliding-window layers (no buffer that
 * grows with the square of the prompt). Sequences are limited to 16,384
 * tokens. Pinned to a commit, so browsers download this version's files.
 */
const LFM3_WEB_ID =
  "daveycodez/LFM3-1.7B-Web@3c75909fbf17af6f734263e9cee75625bcfd7523"
const LFM2_5_ID = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
/** LFM2.5 has no thinking switch: thinking is a separate model. */
const LFM2_5_THINKING_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX"
/**
 * LiquidAI's LFM2.5 q4f16 builds with a `num_logits_to_keep` input, so
 * reading a prompt makes logits for the last token only, not every token
 * (210 MB for an 800-token prompt, which crashed iPhone Safari with web
 * search on). Same weights. Pinned to a commit, like LFM3 Web.
 */
const LFM2_5_WEB_ID =
  "daveycodez/LFM2.5-1.2B-Web@7cbec27ef76aac9be538a16dd5dd57d19328c654"
const LFM2_5_WEB_THINKING_ID =
  "daveycodez/LFM2.5-1.2B-Thinking-Web@eac258e512896325899cf2be2bbd18b546c71706"
const LOCAL_MODELS: Partial<Record<ModelId, string>> = {
  lfm3: LFM3_ID,
  "lfm3-web": LFM3_WEB_ID,
  "lfm2.5": LFM2_5_ID,
  "lfm2.5-web": LFM2_5_WEB_ID
}
/** The model each LFM2.5 option runs instead when Think is on. */
const THINKING_MODELS: Partial<Record<ModelId, string>> = {
  "lfm2.5": LFM2_5_THINKING_ID,
  "lfm2.5-web": LFM2_5_WEB_THINKING_ID
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

/**
 * LFM2.5 Instruct and Thinking, with each model card's sampling. The Think
 * toggle picks the Thinking model when the message is sent.
 */
function lfm25Connection(
  instructId: string,
  thinkingId: string,
  name: string
): ConnectConnectionAdapter {
  const instruct = localModelConnection({
    modelId: instructId,
    name,
    sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.05 },
    toolFormat: "lfm2"
  })
  const thinking = localModelConnection({
    modelId: thinkingId,
    name: `${name} Thinking`,
    sampling: { temperature: 0.05, top_k: 50, repetition_penalty: 1.05 },
    toolFormat: "lfm2",
    // Always thinks: give it the longer thinking budget.
    thinking: () => true
  })
  return {
    connect: (...args) =>
      (thinkingStore.get() ? thinking : instruct).connect(...args)
  }
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
    connection: lfm25Connection(LFM2_5_ID, LFM2_5_THINKING_ID, "LFM2.5")
  },
  {
    id: "lfm2.5-web",
    name: "LFM2.5 1.2B Web",
    icon: "/icons/liquid.svg",
    connection: lfm25Connection(
      LFM2_5_WEB_ID,
      LFM2_5_WEB_THINKING_ID,
      "LFM2.5 Web"
    )
  }
]

/** The Hugging Face model to run for `id`, given the Think toggle. */
const localModelFor = (id: ModelId, thinking: boolean) =>
  (thinking && THINKING_MODELS[id]) || LOCAL_MODELS[id]

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
