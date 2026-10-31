/**
 * The model the user picked, and how hard it thinks. Each session sends
 * both with every run in `forwardedProps` (see the Thread component), and
 * the run picks its adapter and reasoning level from them
 * (./connection.ts). Both are saved in the settings collection.
 */
import { REASONING_LEVELS } from "@tanstack/ai"
import type { ReasoningLevel } from "@tanstack/ai"

import { setting } from "#/lib/settings"
import {
  DEFAULT_MODEL,
  hasWebGpuNow,
  isApplePlatform,
  offered,
  useHasBuiltInModel,
  useHasWebGl2,
  useHasWebGpu,
  useIsApplePlatform
} from "./models"
import type { ModelId } from "./models"

const modelSetting = setting("model", DEFAULT_MODEL)

/**
 * How hard models think, one level for every model: each run clamps it to
 * the model's own levels. High by default: LFM3 answers much better
 * thinking.
 */
const effortSetting = setting("effort", "high")

/** The picked model, if this device can run it, else the default for it. */
export const selectedModel = () =>
  offered(modelSetting.get(), isApplePlatform(), hasWebGpuNow())

export const useSelectedModel = () => {
  // The built-in model and the WebGL models are offered once the browser's
  // checks are back.
  useHasBuiltInModel()
  useHasWebGl2()
  return offered(modelSetting.use(), useIsApplePlatform(), useHasWebGpu())
}

export const setSelectedModel = (id: ModelId) => modelSetting.set(id)

export const useEffort = effortSetting.use

export const setEffort = (level: ReasoningLevel) => effortSetting.set(level)

/** Whether `value` is a reasoning level, as a run reads it. */
export const isReasoningLevel = (value: unknown): value is ReasoningLevel =>
  REASONING_LEVELS.includes(value as ReasoningLevel)

/** What each run of a session forwards, for the run to read. */
export type ChatForwardedProps = { model: ModelId; effort: ReasoningLevel }
