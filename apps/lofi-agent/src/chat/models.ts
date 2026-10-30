import { clampReasoningLevel, supportedReasoningLevels } from "@tanstack/ai"
import type { ModelReasoning, ReasoningLevel } from "@tanstack/ai"
import {
  hasWebGpu,
  modelStatus,
  runsWithoutWebGpu,
  TRANSFORMERS_JS_MODEL_REASONING,
  transformersJsText
} from "@tanstack-ai-browser/transformers-js"
import type {
  ModelStatus,
  TransformersJsListedModel
} from "@tanstack-ai-browser/transformers-js"
import {
  APFEL_BASE_URL,
  apfelText,
  isApfelAvailable
} from "@tanstack-ai-browser/apfel"
import { useSyncExternalStore } from "react"

import { createStore } from "#/lib/store"

export type ModelId =
  | "apple-fm"
  | "gemma-4-e4b"
  | "lfm3"
  | "lfm3-web"
  | "lfm2.5"
  | "lfm2.5-thinking"
  | "lfm2.5-web"
  | "lfm2.5-thinking-web"
  | "lfm2-tool"

/** Where the app reaches apfel on this Mac: its default address. */
const APFEL_URL = APFEL_BASE_URL
const APPLE_MODEL = "apple-foundationmodel"

/** A Transformers.js model of the package's list, in the picker. */
const local = (
  id: ModelId,
  name: string,
  model: TransformersJsListedModel
) => ({
  id,
  name,
  icon: model.includes("gemma") ? "/icons/gemma.svg" : "/icons/liquid.svg",
  adapter: transformersJsText(model, { debug: import.meta.env.DEV }),
  /** The model it downloads, for its loading status. */
  download: model as string | null,
  /** No CPU weights: it can't run where there's no WebGPU. */
  needsWebGpu: !runsWithoutWebGpu(model),
  reasoning: TRANSFORMERS_JS_MODEL_REASONING[model] as
    ModelReasoning | undefined
})

export const MODELS = [
  {
    id: "apple-fm",
    name: "Apple Foundation Model",
    /** Provider logo in /public/icons (LobeHub icons, MIT). */
    icon: "/icons/apple.svg",
    adapter: apfelText(APPLE_MODEL, { baseURL: APFEL_URL }),
    /** Runs outside the browser: nothing to download. */
    download: null,
    needsWebGpu: false,
    reasoning: undefined
  },
  local("gemma-4-e4b", "Gemma 4 E4B", "onnx-community/gemma-4-E4B-it-ONNX"),
  local("lfm3", "LFM3 1.7B", "onnx-community/LFM3-1.7B-ONNX"),
  local("lfm3-web", "LFM3 1.7B Web", "daveycodez/LFM3-1.7B-Web"),
  local("lfm2.5", "LFM2.5 1.2B", "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"),
  local(
    "lfm2.5-thinking",
    "LFM2.5 1.2B Thinking",
    "LiquidAI/LFM2.5-1.2B-Thinking-ONNX"
  ),
  local("lfm2.5-web", "LFM2.5 1.2B Web", "daveycodez/LFM2.5-1.2B-Web"),
  local(
    "lfm2.5-thinking-web",
    "LFM2.5 1.2B Thinking Web",
    "daveycodez/LFM2.5-1.2B-Thinking-Web"
  ),
  local("lfm2-tool", "LFM2 1.2B Tool", "onnx-community/LFM2-1.2B-Tool-ONNX")
] satisfies Array<{
  id: ModelId
  name: string
  icon: string
  adapter: ReturnType<typeof transformersJsText> | ReturnType<typeof apfelText>
  download: string | null
  needsWebGpu: boolean
  reasoning: ModelReasoning | undefined
}>

/**
 * The picker's sections: each adapter's provider (its `name`), in order,
 * with the name shown as the section's heading.
 */
export const PROVIDERS = [
  { id: "apfel", name: "apfel" },
  { id: "transformers-js", name: "Transformers.js" }
] as const

/**
 * The effort levels to offer for a model, least to most thinking: the
 * levels its reasoning data has (LFM3 and Gemma 4: off and high). None for
 * a model with one level or none, which has nothing to pick: LFM2.5
 * Thinking always thinks, the others never do.
 */
export function effortLevels(id: ModelId): ReadonlyArray<ReasoningLevel> {
  const levels = supportedReasoningLevels(modelInfo(id).reasoning)
  return levels.length > 1 ? levels : []
}

/**
 * The level a model runs at for the effort picked: that level if the model
 * has it, else the nearest it has (TanStack AI's clamp). Undefined for a
 * model that doesn't reason.
 */
export const effortFor = (id: ModelId, effort: ReasoningLevel) => {
  const { reasoning } = modelInfo(id)
  return reasoning ? clampReasoningLevel(reasoning, effort) : undefined
}

/** Each level's name in the picker. */
const EFFORT_LABELS: Record<ReasoningLevel, string> = {
  off: "Off",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra high",
  max: "Max"
}

/**
 * Whether a model's thinking is only on or off: "off" and one level (LFM3
 * and Gemma 4 take a true/false switch, declared as off and high).
 */
const isThinkingSwitch = (levels: ReadonlyArray<ReasoningLevel>) =>
  levels.length === 2 && levels[0] === "off"

/**
 * The picker's options for a model's levels: Off and On for a model that
 * only switches thinking, else each level by name.
 */
export function effortOptions(id: ModelId) {
  const levels = effortLevels(id)
  const isSwitch = isThinkingSwitch(levels)
  return levels.map((level) => ({
    id: level,
    name: isSwitch ? (level === "off" ? "Off" : "On") : EFFORT_LABELS[level]
  }))
}

/**
 * What the picker's button says after the model's name: "Thinking" while a
 * switch-only model thinks, the level for a model with levels, nothing when
 * it doesn't think or has nothing to pick.
 */
export function effortButtonLabel(id: ModelId, level: ReasoningLevel) {
  const levels = effortLevels(id)
  if (levels.length === 0 || level === "off") return undefined
  return isThinkingSwitch(levels) ? "Thinking" : EFFORT_LABELS[level]
}

/**
 * The model to start with, and to fall back on: LFM3 Web runs anywhere,
 * iPhone Safari included, thinking at high effort (see ./model.ts).
 */
export const DEFAULT_MODEL: ModelId = "lfm3-web"

/**
 * The model to start with where there's no WebGPU (Firefox on Android, for
 * one), and models run on the CPU: LFM2 1.2B Tool, which has CPU weights
 * ONNX Runtime's WebAssembly build can run, and is made for tool calls.
 */
export const DEFAULT_CPU_MODEL: ModelId = "lfm2-tool"

export const modelInfo = (id: ModelId) =>
  MODELS.find((m) => m.id === id) ??
  MODELS.find((m) => m.id === DEFAULT_MODEL) ??
  MODELS[0]

/** Whether `value` is the id of a model in the list, as a run reads it. */
export const isModelId = (value: unknown): value is ModelId =>
  MODELS.some((m) => m.id === value)

/** Apple's model only exists on macOS and iOS (iPadOS reports itself as a Mac). */
export const isApplePlatform = () =>
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad|iPod/.test(navigator.userAgent)

const noopSubscribe = () => () => undefined
export const useIsApplePlatform = () =>
  useSyncExternalStore(noopSubscribe, isApplePlatform, () => false)

/**
 * Whether models run on WebGPU here. Until the GPU check is back, what
 * `navigator` says (Firefox on Android has no `navigator.gpu`); on the
 * server, yes. Without it, only models with CPU weights are offered.
 */
const webGpu = createStore(
  typeof navigator === "undefined" || "gpu" in navigator
)
if (typeof window !== "undefined") void hasWebGpu().then(webGpu.set)

export const hasWebGpuNow = webGpu.get

export const useHasWebGpu = () =>
  useSyncExternalStore(webGpu.subscribe, webGpu.get, () => true)

/** What this device can run: Apple's model on Apple devices, CPU models without WebGPU. */
const canRun = (id: ModelId, isApple: boolean, hasGpu: boolean) =>
  isModelId(id) &&
  (id !== "apple-fm" || isApple) &&
  (hasGpu || !modelInfo(id).needsWebGpu)

/**
 * The pick, if this device can run it, else the default: LFM3 Web, or
 * LFM2 1.2B Tool where there's no WebGPU.
 */
export const offered = (
  id: ModelId,
  isApple: boolean,
  hasGpu: boolean
): ModelId =>
  canRun(id, isApple, hasGpu) ? id : hasGpu ? DEFAULT_MODEL : DEFAULT_CPU_MODEL

/** Models to offer in the picker on this device. */
export function useModels() {
  const isApple = useIsApplePlatform()
  const hasGpu = useHasWebGpu()
  return MODELS.filter((m) => canRun(m.id, isApple, hasGpu))
}

const IDLE = createStore<ModelStatus>({ state: "idle" })

/** Loading status of the model `id` downloads, if it downloads one. */
export function useModelStatus(id: ModelId): ModelStatus {
  const download = modelInfo(id).download
  const store = download ? modelStatus(download) : IDLE
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
export const checkAfm = () =>
  isApfelAvailable(APPLE_MODEL, { baseURL: APFEL_URL })
