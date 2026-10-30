import { Button, Disclosure, Spinner } from "@heroui/react"
import type { ReactNode } from "react"

import { Markdown } from "#/components/ai/markdown"

/** A model's thinking, collapsed under a toggle. */
export function Reasoning({ children }: { children: ReactNode }) {
  return <Disclosure>{children}</Disclosure>
}

export function ReasoningTrigger({ isStreaming }: { isStreaming: boolean }) {
  return (
    <Disclosure.Heading>
      <Button
        slot="trigger"
        size="sm"
        variant="ghost"
        className="-ms-3 text-muted"
      >
        {isStreaming && <Spinner size="sm" color="current" />}
        {isStreaming ? "Thinking…" : "Thoughts"}
        {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
        <Disclosure.Indicator className="ms-0" />
      </Button>
    </Disclosure.Heading>
  )
}

export function ReasoningContent({ children }: { children: string }) {
  return (
    <Disclosure.Content>
      <Disclosure.Body className="markdown prose prose-sm max-w-none border-s-2 border-separator ps-3 text-muted wrap-break-word">
        <Markdown>{children}</Markdown>
      </Disclosure.Body>
    </Disclosure.Content>
  )
}
