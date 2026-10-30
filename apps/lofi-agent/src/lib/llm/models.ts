import {
  modelStatus,
  transformersJs
} from "@tanstack-ai-browser/transformers-js"
import type {
  ModelStatus,
  RunStats,
  Sampling,
  TransformersJsOptions
} from "@tanstack-ai-browser/transformers-js"
import { apfel, isApfelAvailable } from "@tanstack-ai-browser/apfel"
import { useSyncExternalStore } from "react"

import { recordGeneration } from "#/lib/llm/generation-stats"
import type { GenerationStats } from "#/lib/llm/generation-stats"
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

/** A model run's counts and timings, as the stats line takes them. */
function generationStats({
  inputTokens,
  cachedTokens,
  outputTokens,
  firstTokenMs,
  promptMs,
  decodeMs
}: RunStats): GenerationStats {
  return {
    inputTokens,
    ...(cachedTokens > 0 && { cachedTokens }),
    ...(promptMs !== null && { promptMs }),
    outputTokens,
    ...(firstTokenMs !== null && {
      firstTokenMs,
      decodeTokens: Math.max(outputTokens - 1, 0),
      decodeMs
    })
  }
}

/** Each model call's counts go to the stats line, as it runs and at the end. */
const onStats: TransformersJsOptions["onStats"] = (stats, run) =>
  recordGeneration(run.threadId, run.callId, generationStats(stats), run.done)

/** LFM3 card, "General" (chat, tool use, agents, coding). */
const LFM3_SAMPLING: Sampling = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.1
}
const LFM2_5_SAMPLING: Sampling = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.05
}
const LFM2_5_THINKING_SAMPLING: Sampling = {
  temperature: 0.05,
  top_k: 50,
  repetition_penalty: 1.05
}

const lfm3 = (modelId: string) =>
  transformersJs(modelId, {
    toolFormat: "lfm3",
    sampling: LFM3_SAMPLING,
    onStats,
    debug: import.meta.env.DEV
  })

/** LFM2.5 Instruct, and its Thinking model when Think is on. */
const lfm25 = (instructId: string, thinkingId: string) =>
  transformersJs(instructId, {
    toolFormat: "lfm2",
    sampling: LFM2_5_SAMPLING,
    thinking: { model: thinkingId, sampling: LFM2_5_THINKING_SAMPLING },
    onStats,
    debug: import.meta.env.DEV
  })

/** Apple's on-device model, through apfel on this Mac. */
const appleFoundationModel = apfel({
  onStats: (stats, run) =>
    recordGeneration(run.threadId, run.callId, stats, run.done)
})

export const MODELS = [
  {
    id: "apple-fm",
    name: "Apple Foundation Model",
    /** Provider logo in /public/icons (LobeHub icons, MIT). */
    icon: "/icons/apple.svg",
    adapter: appleFoundationModel,
    /** Runs outside the browser: no download to report. */
    runs: null
  },
  {
    id: "lfm3",
    name: "LFM3 1.7B",
    icon: "/icons/liquid.svg",
    adapter: lfm3(LFM3_ID),
    /** The Hugging Face model it runs, with Think off and on. */
    runs: { off: LFM3_ID, on: LFM3_ID }
  },
  {
    id: "lfm3-web",
    name: "LFM3 1.7B Web",
    icon: "/icons/liquid.svg",
    adapter: lfm3(LFM3_WEB_ID),
    runs: { off: LFM3_WEB_ID, on: LFM3_WEB_ID }
  },
  {
    id: "lfm2.5",
    name: "LFM2.5 1.2B",
    icon: "/icons/liquid.svg",
    adapter: lfm25(LFM2_5_ID, LFM2_5_THINKING_ID),
    runs: { off: LFM2_5_ID, on: LFM2_5_THINKING_ID }
  },
  {
    id: "lfm2.5-web",
    name: "LFM2.5 1.2B Web",
    icon: "/icons/liquid.svg",
    adapter: lfm25(LFM2_5_WEB_ID, LFM2_5_WEB_THINKING_ID),
    runs: { off: LFM2_5_WEB_ID, on: LFM2_5_WEB_THINKING_ID }
  }
] satisfies Array<{
  id: ModelId
  name: string
  icon: string
  adapter: ReturnType<typeof transformersJs> | ReturnType<typeof apfel>
  runs: { off: string; on: string } | null
}>

/**
 * The model to start with, and to fall back on: LFM3 Web runs anywhere,
 * iPhone Safari included, with thinking on (see `thinkingStore`).
 */
export const DEFAULT_MODEL: ModelId = "lfm3-web"

export const modelInfo = (id: ModelId) =>
  MODELS.find((m) => m.id === id) ??
  MODELS.find((m) => m.id === DEFAULT_MODEL) ??
  MODELS[0]

/** The model the user picked; saved only once they pick one. */
export const modelStore = createStore<ModelId>(
  DEFAULT_MODEL,
  "lofi-agent:model"
)

/** Apple's model only exists on macOS and iOS (iPadOS reports itself as a Mac). */
const isApplePlatform = () =>
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad|iPod/.test(navigator.userAgent)

const noopSubscribe = () => () => undefined
const useIsApplePlatform = () =>
  useSyncExternalStore(noopSubscribe, isApplePlatform, () => false)

/**
 * The pick, if this device offers it: the Apple model only on Apple
 * devices, and a model no longer offered falls back to the default.
 */
const offered = (id: ModelId, isApple: boolean): ModelId =>
  MODELS.some((m) => m.id === id) && (id !== "apple-fm" || isApple)
    ? id
    : DEFAULT_MODEL

export const currentModel = () => offered(modelStore.get(), isApplePlatform())
export const useCurrentModel = () =>
  offered(modelStore.use(), useIsApplePlatform())

/** Models to offer in the picker on this device. */
export function useModels() {
  const isApple = useIsApplePlatform()
  return MODELS.filter((m) => m.id !== "apple-fm" || isApple)
}

const IDLE = createStore<ModelStatus>({ state: "idle" })

/** Loading status of the model the Think toggle would run for `id`. */
export function useModelStatus(id: ModelId): ModelStatus {
  const runs = modelInfo(id).runs
  const thinking = thinkingStore.use()
  const status = runs ? modelStatus(thinking ? runs.on : runs.off) : null
  const store = status ?? { subscribe: IDLE.subscribe, get: IDLE.get }
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}

/** Result of the last Apple model check, shown next to the picker. */
export const afmStatus = createStore<"idle" | "checking" | "unavailable">(
  "idle"
)

/**
 * Whether Apple's model is reachable through apfel. Reaching localhost from
 * a website makes Chrome ask for permission, so this only runs when the
 * user picks the Apple model.
 */
export const checkAfm = () => isApfelAvailable()
