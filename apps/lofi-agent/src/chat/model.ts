/**
 * The model the user picked, and whether it thinks first. Each session sends
 * both with every run in `forwardedProps` (see the Thread component), and
 * the run picks its adapter from them (./connection.ts). Both are saved in
 * the settings collection.
 */
import { setting } from "#/lib/settings"
import {
  DEFAULT_MODEL,
  isApplePlatform,
  offered,
  useIsApplePlatform
} from "./models"
import type { ModelId } from "./models"

const modelSetting = setting("model", DEFAULT_MODEL)

/**
 * Whether models with a thinking switch reason before answering. On by
 * default: LFM3 answers much better with it.
 */
const thinkingSetting = setting("think", true)

/** The picked model, if this device offers it, else the default. */
export const selectedModel = () =>
  offered(modelSetting.get(), isApplePlatform())

export const useSelectedModel = () =>
  offered(modelSetting.use(), useIsApplePlatform())

export const setSelectedModel = (id: ModelId) => modelSetting.set(id)

export const useThinking = thinkingSetting.use

export const setThinking = (on: boolean) => thinkingSetting.set(on)

/** What each run of a session forwards, for the run to read. */
export type ChatForwardedProps = { model: ModelId; think: boolean }
