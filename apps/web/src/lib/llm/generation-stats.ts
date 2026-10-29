/**
 * Token counts and timings of each run of the model, by chat, for the stats
 * line under the composer. Connections record a run each time the model reads
 * a prompt and generates; a reply that calls tools runs it more than once.
 */

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
  /** Tokens generated after the first one. */
  decodeTokens?: number
  /** From the first generated token to the last, in ms. */
  decodeMs?: number
  /** Run by a sub-agent of the reply rather than the chat's own agent. */
  subAgent?: boolean
}

const runs = new Map<string, Array<GenerationStats>>()

export function recordGeneration(threadId: string, stats: GenerationStats) {
  const list = runs.get(threadId)
  if (list) list.push(stats)
  else runs.set(threadId, [stats])
}

/** Removes and returns a chat's recorded runs. */
export function takeGenerations(threadId: string) {
  const list = runs.get(threadId) ?? []
  runs.delete(threadId)
  return list
}

/** The runs recorded for a chat since it was last cleared. */
export const generationsOf = (threadId: string) => runs.get(threadId) ?? []

export const clearGenerations = (threadId: string) => runs.delete(threadId)

/** What the stats line shows for a reply. */
export type ReplyStats = {
  inputTokens?: number
  cachedTokens?: number
  outputTokens?: number
  totalTokens?: number
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
  return {
    inputTokens: last?.inputTokens,
    cachedTokens: last?.cachedTokens,
    outputTokens: sum(generations.map((g) => g.outputTokens)),
    totalTokens: lastTotal,
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

/** "Input 868 · Cached 512 · Output 214 · Total 1,082 · 18.4 tok/s · TTFT 2.1s" */
export function formatReplyStats(stats: ReplyStats) {
  const n = (value: number | undefined, label: string) =>
    value !== undefined ? `${label} ${count.format(value)}` : null
  return [
    n(stats.inputTokens, "Input"),
    n(stats.cachedTokens, "Cached"),
    n(stats.outputTokens, "Output"),
    n(stats.totalTokens, "Total"),
    stats.tokensPerSecond !== undefined &&
      `${oneDecimal.format(stats.tokensPerSecond)} tok/s`,
    stats.firstTokenSeconds !== undefined &&
      `TTFT ${oneDecimal.format(stats.firstTokenSeconds)}s`
  ]
    .filter(Boolean)
    .join(" · ")
}
