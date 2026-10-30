import { Tooltip } from "@heroui/react"
import { useSyncExternalStore } from "react"

/** One clock for every "time ago": ticks every 30 seconds while shown. */
let now = Date.now()
const listeners = new Set<() => void>()
let timer: ReturnType<typeof setInterval> | undefined

function subscribe(listener: () => void) {
  listeners.add(listener)
  timer ??= setInterval(() => {
    now = Date.now()
    listeners.forEach((l) => l())
  }, 30_000)
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
  }
}

const getNow = () => now

const relative = new Intl.RelativeTimeFormat("en-US", { numeric: "auto" })

/** "just now", "17 minutes ago", "yesterday", "3 weeks ago". */
function timeAgo(date: Date, at: number) {
  const seconds = Math.round((date.getTime() - at) / 1000)
  if (Math.abs(seconds) < 60) return "just now"
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ["minute", 60],
    ["hour", 60 * 60],
    ["day", 60 * 60 * 24],
    ["week", 60 * 60 * 24 * 7],
    ["month", 60 * 60 * 24 * 30],
    ["year", 60 * 60 * 24 * 365]
  ]
  let unit = units[0]
  for (const candidate of units) {
    if (Math.abs(seconds) >= candidate[1]) unit = candidate
  }
  return relative.format(Math.round(seconds / unit[1]), unit[0])
}

/** "Oct 30, 2026, 12:02 AM". */
const written = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit"
})

/** How long ago `date` was, with the date written out on hover. */
export function TimeAgo({ date }: { date: Date }) {
  const at = useSyncExternalStore(subscribe, getNow, getNow)
  const full = written.format(date)
  return (
    <Tooltip delay={300}>
      {/* A flex box, so the text centers with the buttons beside it. */}
      <Tooltip.Trigger aria-label={full} className="flex cursor-default items-center">
        <span className="px-1 text-xs text-muted tabular-nums">
          {timeAgo(date, Math.max(at, date.getTime()))}
        </span>
      </Tooltip.Trigger>
      <Tooltip.Content>{full}</Tooltip.Content>
    </Tooltip>
  )
}
