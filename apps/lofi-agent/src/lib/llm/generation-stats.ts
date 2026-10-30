/**
 * Token counts and timings of each run of the model, by chat, for the stats
 * line under the composer. Each model call's usage is recorded when it ends
 * (`chat()`'s `onUsage`); a reply that calls tools runs the model more than
 * once. Each chat's latest reply's stats are kept in localStorage, by thread
 * id, until the chat is deleted.
 */
import type { TokenUsage } from "@tanstack/ai"
import type { ApfelProviderUsageDetails } from "@tanstack-ai-browser/apfel"
import type { TransformersJsProviderUsageDetails } from "@tanstack-ai-browser/transformers-js"

import { createStore } from "#/lib/store"

/** One run of the model. Counts the source doesn't report are left out. */
export type GenerationStats = {
  /** Tokens of the prompt the model read (after earlier turns were trimmed). */
  inputTokens?: number
  /** Input tokens served from a cache instead of being read again. */
  cachedTokens?: number
  /** Tokens the model generated. */
  outputTokens?: number
  /** From sending the prompt to the first generated token, in ms. */
  firstTokenMs?: number
  /** From starting to read the prompt to the first generated token, in ms. */
  promptMs?: number
  /** Tokens generated after the first one. */
  decodeTokens?: number
  /** From the first generated token to the last, in ms. */
  decodeMs?: number
}

/** Timings both adapters report, under the same names. */
type UsageTimings = TransformersJsProviderUsageDetails & ApfelProviderUsageDetails

/** A model call's stats from its usage: token counts, and the adapter's timings. */
export function statsFromUsage(usage: TokenUsage): GenerationStats {
  const timings = usage.providerUsageDetails as UsageTimings | undefined
  const cachedTokens = usage.promptTokensDetails?.cachedTokens
  return {
    inputTokens: usage.promptTokens,
    ...(cachedTokens && { cachedTokens }),
    outputTokens: usage.completionTokens,
    ...(timings?.firstTokenMs !== undefined && {
      firstTokenMs: timings.firstTokenMs
    }),
    ...(timings?.promptMs !== undefined && { promptMs: timings.promptMs }),
    ...(timings?.decodeMs !== undefined && {
      decodeMs: timings.decodeMs,
      decodeTokens: timings.decodeTokens ?? Math.max(usage.completionTokens - 1, 0)
    })
  }
}

/** The current reply's runs in each chat, by key, in the order they started. */
const runs = new Map<string, Map<string, GenerationStats>>()

/** Each chat's current reply's stats as they come in (this session). */
const live = createStore<Record<string, ReplyStats | null>>({})

/** Each chat's latest reply's stats, kept across reloads. */
const saved = createStore<Record<string, ReplyStats>>(
  {},
  "lofi-agent:reply-stats"
)

const without = <T>(record: Record<string, T>, ids: Array<string>) =>
  Object.fromEntries(Object.entries(record).filter(([id]) => !ids.includes(id)))

/**
 * Starts a new reply in a chat: its stats line empties until the model's
 * first run ends. Runs after tool calls are part of the same reply.
 */
export function startReply(threadId: string) {
  runs.set(threadId, new Map())
  live.set((all) => ({ ...all, [threadId]: null }))
  if (threadId in saved.get()) saved.set((all) => without(all, [threadId]))
}

/** Records a finished run of the model in a chat's reply, and saves the reply's stats. */
export function recordGeneration(
  threadId: string,
  key: string,
  stats: GenerationStats
) {
  let list = runs.get(threadId)
  if (!list) {
    list = new Map()
    runs.set(threadId, list)
  }
  list.set(key, stats)
  const reply = replyStats([...list.values()])
  live.set((all) => ({ ...all, [threadId]: reply }))
  if (reply) saved.set((all) => ({ ...all, [threadId]: reply }))
}

/** Forgets deleted chats' stats. */
export function forgetReplyStats(threadIds: Array<string>) {
  for (const id of threadIds) runs.delete(id)
  if (threadIds.some((id) => id in live.get())) {
    live.set((all) => without(all, threadIds))
  }
  if (threadIds.some((id) => id in saved.get())) {
    saved.set((all) => without(all, threadIds))
  }
}

/** A chat's current or latest reply's stats, or null if it has none. */
export function latestReplyStats(threadId: string) {
  const current = live.get()
  return threadId in current
    ? current[threadId]
    : (saved.get()[threadId] ?? null)
}

/** Calls `listener` whenever any chat's stats change. */
export function subscribeReplyStats(listener: () => void) {
  const stopLive = live.subscribe(listener)
  const stopSaved = saved.subscribe(listener)
  return () => {
    stopLive()
    stopSaved()
  }
}

/** What the stats line shows for a reply. */
export type ReplyStats = {
  inputTokens?: number
  cachedTokens?: number
  outputTokens?: number
  totalTokens?: number
  /** Prompt tokens read per second, those served from a cache left out. */
  promptTokensPerSecond?: number
  tokensPerSecond?: number
  firstTokenSeconds?: number
}

/** Below this much generating time, a tokens-per-second figure is noise. */
const MIN_DECODE_MS = 100

const sum = (values: Array<number | undefined>) =>
  values.every((v) => v !== undefined)
    ? values.reduce<number>((a, b) => a + (b ?? 0), 0)
    : undefined

/** A reply's stats from its runs, in the order they ran. */
export function replyStats(
  generations: Array<GenerationStats>
): ReplyStats | null {
  if (generations.length === 0) return null
  // The prompt of the last run is what the model last read.
  const last = generations.at(-1)
  const timed = generations.filter(
    (g) => g.decodeTokens !== undefined && g.decodeMs !== undefined
  )
  const decodeTokens = sum(timed.map((g) => g.decodeTokens)) ?? 0
  const decodeMs = sum(timed.map((g) => g.decodeMs)) ?? 0
  const lastTotal =
    last?.inputTokens !== undefined && last.outputTokens !== undefined
      ? last.inputTokens + last.outputTokens
      : undefined
  const firstTokenMs = generations[0].firstTokenMs
  // Prompt reading across the reply's runs: what each read fresh, and how
  // long until its first token.
  const read = generations.filter(
    (g) => g.inputTokens !== undefined && g.promptMs !== undefined
  )
  const readTokens = sum(
    read.map((g) => (g.inputTokens ?? 0) - (g.cachedTokens ?? 0))
  )
  const readMs = sum(read.map((g) => g.promptMs)) ?? 0
  const cached = generations.filter((g) => g.cachedTokens !== undefined)
  return {
    inputTokens: last?.inputTokens,
    // Tokens every run of the reply took from a cache instead of reading.
    cachedTokens:
      cached.length > 0 ? sum(cached.map((g) => g.cachedTokens)) : undefined,
    outputTokens: sum(generations.map((g) => g.outputTokens)),
    totalTokens: lastTotal,
    promptTokensPerSecond:
      readTokens !== undefined && readTokens > 0 && readMs >= MIN_DECODE_MS
        ? readTokens / (readMs / 1000)
        : undefined,
    tokensPerSecond:
      decodeMs >= MIN_DECODE_MS ? decodeTokens / (decodeMs / 1000) : undefined,
    firstTokenSeconds:
      firstTokenMs !== undefined ? firstTokenMs / 1000 : undefined
  }
}

const count = new Intl.NumberFormat("en-US")
const oneDecimal = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1
})

/** ["In 868", "Cached 512", "Out 214", "Total 1,082", "95.2 pp/s", "18.4 tok/s", "TTFT 2.1s"] */
export function replyStatsItems(stats: ReplyStats) {
  const n = (value: number | undefined, label: string) =>
    value !== undefined ? `${label} ${count.format(value)}` : null
  return [
    n(stats.inputTokens, "In"),
    n(stats.cachedTokens, "Cached"),
    n(stats.outputTokens, "Out"),
    n(stats.totalTokens, "Total"),
    stats.promptTokensPerSecond !== undefined
      ? `${oneDecimal.format(stats.promptTokensPerSecond)} pp/s`
      : null,
    stats.tokensPerSecond !== undefined
      ? `${oneDecimal.format(stats.tokensPerSecond)} tok/s`
      : null,
    stats.firstTokenSeconds !== undefined
      ? `TTFT ${oneDecimal.format(stats.firstTokenSeconds)}s`
      : null
  ].filter((item) => item !== null)
}

/** "In 868 · Cached 512 · Out 214 · Total 1,082 · 95.2 pp/s · 18.4 tok/s · TTFT 2.1s" */
export const formatReplyStats = (stats: ReplyStats) =>
  replyStatsItems(stats).join(" · ")
