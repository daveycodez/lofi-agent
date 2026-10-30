/**
 * Token counts and timings of each run of the model, by chat, for the stats
 * line under the composer. Connections record a run each time the model reads
 * a prompt and generates, as it goes and when it ends; a reply that calls
 * tools runs it more than once. Each chat's latest reply's stats are kept in
 * localStorage, by thread id, until the chat is deleted.
 */
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
  /** Run by a sub-agent of the reply rather than the chat's own agent. */
  subAgent?: boolean
}

/** The current reply's runs in each chat, by key, in the order they started. */
const runs = new Map<string, Map<string, GenerationStats>>()

/** Pseudo-chats whose runs count as sub-agent runs of a chat's reply. */
const subAgentOf = new Map<string, string>()

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
 * Starts a new reply in a chat: its stats line empties until the model
 * reports on its first run. Runs after tool calls are part of the same reply.
 */
export function startReply(threadId: string) {
  if (subAgentOf.has(threadId)) return
  runs.set(threadId, new Map())
  live.set((all) => ({ ...all, [threadId]: null }))
  if (threadId in saved.get()) saved.set((all) => without(all, [threadId]))
}

/**
 * Records a run of the model in a chat's reply. A run reports under the same
 * key as it goes (`done` false), then once more when it ends; the reply's
 * stats are saved when one of its runs ends.
 */
export function recordGeneration(
  threadId: string,
  key: string,
  stats: GenerationStats,
  done = true
) {
  const parent = subAgentOf.get(threadId)
  if (parent) {
    recordGeneration(parent, key, { ...stats, subAgent: true }, done)
    return
  }
  let list = runs.get(threadId)
  if (!list) {
    list = new Map()
    runs.set(threadId, list)
  }
  list.set(key, stats)
  const reply = replyStats([...list.values()])
  live.set((all) => ({ ...all, [threadId]: reply }))
  if (done && reply) saved.set((all) => ({ ...all, [threadId]: reply }))
}

/**
 * Counts the runs recorded under `runId` (a sub-agent's own pseudo-chat) as
 * sub-agent runs of `threadId`'s reply, until the returned function is called.
 */
export function countAsSubAgent(runId: string, threadId: string) {
  subAgentOf.set(runId, threadId)
  return () => {
    subAgentOf.delete(runId)
  }
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
  // The prompt of the agent's last run is what the model last read.
  const last =
    generations.filter((g) => !g.subAgent).at(-1) ?? generations.at(-1)
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
  // Prompt reading across the reply's runs, sub-agents' included: what each
  // read fresh, and how long until its first token.
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
