import { eq, useLiveQuery, useOptionalDbClient } from "@tanstack/react-db"
import { Fragment } from "react"

import { replyStatsCollection } from "#/lib/db"
import { formatReplyStats, replyStatsItems } from "#/chat/generation-stats"

/**
 * One muted line under the composer with the chat's latest reply's stats,
 * updated as the reply runs and kept across reloads. It always takes at
 * least a line, so nothing moves when the stats appear; on a narrow screen
 * it wraps between items.
 */
export function ReplyStatsLine({ threadId }: { threadId: string }) {
  const db = useOptionalDbClient()
  const { data } = useLiveQuery({
    query: (q) =>
      db
        ? q
            .from({ record: replyStatsCollection })
            .where(({ record }) => eq(record.id, threadId))
            .findOne()
        : null
  })
  const stats = data?.stats
  const items = stats ? replyStatsItems(stats) : []
  return (
    // Not a live region: it changes several times during a reply.
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
