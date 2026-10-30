import { Disclosure } from "@heroui/react"
import { createContext, useContext } from "react"
import type { ReactNode } from "react"

import { Markdown } from "#/components/ai/markdown"

const StreamingContext = createContext(false)

/**
 * A model's thinking under a "Thoughts" toggle, collapsed until opened.
 * While it thinks, the toggle reads "Thinking…".
 */
export function Reasoning({
  isStreaming = false,
  children
}: {
  isStreaming?: boolean
  children: ReactNode
}) {
  return (
    <StreamingContext.Provider value={isStreaming}>
      <Disclosure>{children}</Disclosure>
    </StreamingContext.Provider>
  )
}

/** "Thinking…" with a shimmer while it thinks, then "Thoughts". */
export function ReasoningTrigger() {
  const isStreaming = useContext(StreamingContext)
  const label = isStreaming ? "Thinking…" : "Thoughts"
  return (
    <Disclosure.Heading>
      {/* The sources toggle's design at text-sm: only as wide as its label
          and chevron, and only the text color changes on hover. */}
      <Disclosure.Trigger className="flex min-h-8 w-fit items-center gap-2 py-1 text-sm text-muted transition-colors hover:text-foreground">
        <span className={isStreaming ? "text-shimmer" : undefined}>
          {label}
        </span>
        <Disclosure.Indicator className="ms-0 size-3.5 shrink-0 text-muted" />
      </Disclosure.Trigger>
    </Disclosure.Heading>
  )
}

/**
 * The thinking, muted, set off by a line down its left side, in HeroUI's
 * default body padding (as HeroUI Pro's ChainOfThought steps are).
 */
export function ReasoningContent({ children }: { children: string }) {
  return (
    <Disclosure.Content>
      <Disclosure.Body>
        <div className="markdown markdown-muted prose prose-sm max-w-none border-s border-border ps-3 wrap-break-word">
          <Markdown>{children}</Markdown>
        </div>
      </Disclosure.Body>
    </Disclosure.Content>
  )
}
