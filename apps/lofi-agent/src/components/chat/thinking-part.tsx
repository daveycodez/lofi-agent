import type { PartProps } from "@tanstack/ai-react/ui"
import { useContext } from "react"

import type { chatOptions } from "#/chat/options"
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger
} from "#/components/ai/reasoning"
import { ReplyContext } from "./thread-context"

export function ThinkingPart({
  part
}: PartProps<typeof chatOptions, "thinking">) {
  const { isRunning, hasAnswer } = useContext(ReplyContext)
  const thinking = part.content.trim()
  if (!thinking) return null
  return (
    <Reasoning isStreaming={isRunning && !hasAnswer}>
      <ReasoningTrigger />
      <ReasoningContent>{thinking}</ReasoningContent>
    </Reasoning>
  )
}
