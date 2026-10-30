import Bulb from "@gravity-ui/icons/Bulb"
import { Button, Card, Disclosure, Spinner } from "@heroui/react"
import { createContext, useContext, useState } from "react"
import type { ReactNode } from "react"

import { Markdown } from "#/components/ai/markdown"

const StreamingContext = createContext(false)

/**
 * A model's thinking, in a card that opens while it thinks and closes when
 * it's done (as TanStack AI's `ThinkingPart` does). It can be opened again.
 */
export function Reasoning({
  isStreaming = false,
  children
}: {
  isStreaming?: boolean
  children: ReactNode
}) {
  const [isExpanded, setIsExpanded] = useState(isStreaming)
  // Follow the stream: open when thinking starts, close when it ends.
  const [wasStreaming, setWasStreaming] = useState(isStreaming)
  if (isStreaming !== wasStreaming) {
    setWasStreaming(isStreaming)
    setIsExpanded(isStreaming)
  }
  return (
    <StreamingContext.Provider value={isStreaming}>
      <Card className="gap-0 p-0">
        <Disclosure isExpanded={isExpanded} onExpandedChange={setIsExpanded}>
          {children}
        </Disclosure>
      </Card>
    </StreamingContext.Provider>
  )
}

export function ReasoningTrigger() {
  const isStreaming = useContext(StreamingContext)
  return (
    <Disclosure.Heading>
      <Button
        slot="trigger"
        variant="ghost"
        size="sm"
        fullWidth
        className="justify-start rounded-3xl px-3 text-muted"
      >
        {isStreaming ? <Spinner size="sm" color="current" /> : <Bulb />}
        {isStreaming ? "Thinking…" : "Thoughts"}
        <Disclosure.Indicator className="ms-auto" />
      </Button>
    </Disclosure.Heading>
  )
}

export function ReasoningContent({ children }: { children: string }) {
  return (
    <Disclosure.Content>
      <Disclosure.Body className="markdown prose prose-sm max-w-none px-3 pt-0 pb-3 text-muted wrap-break-word">
        <Markdown>{children}</Markdown>
      </Disclosure.Body>
    </Disclosure.Content>
  )
}
