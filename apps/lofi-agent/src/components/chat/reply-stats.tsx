import { Fragment, useSyncExternalStore } from "react"

import {
  formatReplyStats,
  latestReplyStats,
  replyStatsItems,
  subscribeReplyStats
} from "#/lib/llm/generation-stats"

/**
 * One muted line under the composer with the chat's latest reply's stats,
 * updated as the reply streams and kept across reloads. It always takes at
 * least a line, so nothing moves when the stats appear; on a narrow screen
 * it wraps between items.
 */
export function ReplyStatsLine({ threadId }: { threadId: string }) {
  const stats = useSyncExternalStore(
    subscribeReplyStats,
    () => latestReplyStats(threadId),
    () => null
  )
  const items = stats ? replyStatsItems(stats) : []
  return (
    // Not a live region: it changes several times a second while a reply
    // streams.
    <p
      title={stats ? formatReplyStats(stats) : undefined}
      className="mt-1.5 min-h-4 text-center text-xs/4 text-muted tabular-nums"
    >
      {items.map((item, i) => (
        <Fragment key={item.replace(/[\d.,]+/g, "#")}>
          {i > 0 && " · "}
          <span className="whitespace-nowrap">{item}</span>
        </Fragment>
      ))}
    </p>
  )
}
