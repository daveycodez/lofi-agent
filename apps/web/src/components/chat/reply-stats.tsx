import type { UIMessage } from "@tanstack/ai"
import { useEffect, useRef, useState } from "react"

import {
  clearGenerations,
  formatReplyStats,
  generationsOf,
  replyStats
} from "#/lib/llm/generation-stats"
import type { ReplyStats } from "#/lib/llm/generation-stats"

/**
 * The latest finished reply's stats in this chat (this session only), or
 * null while a reply runs or before the first one. A reply starts when a run
 * starts after the user's message; runs that go on after tool calls are part
 * of the same reply.
 */
export function useReplyStats(
  threadId: string,
  isLoading: boolean,
  messages: ReadonlyArray<UIMessage>
) {
  const [latest, setLatest] = useState<{
    threadId: string
    stats: ReplyStats | null
  } | null>(null)
  const wasLoading = useRef(false)
  const lastIsUser = messages.at(-1)?.role === "user"
  useEffect(() => {
    if (isLoading && !wasLoading.current && lastIsUser) {
      clearGenerations(threadId)
    }
    if (!isLoading && wasLoading.current) {
      setLatest({ threadId, stats: replyStats(generationsOf(threadId)) })
    }
    wasLoading.current = isLoading
  }, [isLoading, lastIsUser, threadId])
  return !isLoading && latest?.threadId === threadId ? latest.stats : null
}

/**
 * One muted line under the composer. It always takes its line, so nothing
 * moves when the stats appear.
 */
export function ReplyStatsLine({ stats }: { stats: ReplyStats | null }) {
  const text = stats ? formatReplyStats(stats) : ""
  return (
    <p
      aria-live="polite"
      title={text || undefined}
      className="mt-1.5 h-4 truncate text-center text-xs text-muted tabular-nums"
    >
      {text}
    </p>
  )
}
