import { memo } from "react"
import type { HTMLAttributes } from "react"

import { Markdown } from "#/components/ai/markdown"
import type { Source } from "#/lib/sources"

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: "user" | "assistant"
}

/**
 * One message. A user message sits on the right, its actions beside it; an
 * assistant message stacks its parts, then its sources and actions.
 */
export function Message({ from, className = "", ...props }: MessageProps) {
  return (
    <div
      data-role={from}
      className={`group ${
        from === "user"
          ? "flex items-end justify-end gap-2"
          : "flex flex-col gap-2"
      } ${className}`}
      {...props}
    />
  )
}

/** A user message's text, in a bubble. */
export function MessageContent({
  className = "",
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`max-w-[80%] rounded-2xl bg-default px-4 py-2 wrap-break-word whitespace-pre-wrap text-default-foreground ${className}`}
      {...props}
    />
  )
}

/** An assistant's answer, as Markdown, with citation chips for `sources`. */
export const MessageResponse = memo(function MessageResponse({
  children,
  sources
}: {
  children: string
  /** What `[n]` in the text refers to. */
  sources?: Array<Source>
}) {
  return (
    <div className="markdown prose max-w-none wrap-break-word">
      <Markdown sources={sources}>{children}</Markdown>
    </div>
  )
})
