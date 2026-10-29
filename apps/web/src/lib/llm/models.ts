import type { ConnectConnectionAdapter } from "@tanstack/ai-react"
import { useSyncExternalStore } from "react"

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
