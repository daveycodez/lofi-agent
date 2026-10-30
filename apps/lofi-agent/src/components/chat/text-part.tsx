import type { PartProps } from "@tanstack/ai-react/ui"
import { useContext } from "react"

import type { chatOptions } from "#/chat/options"
import { MessageResponse } from "#/components/ai/message"
import { ReplyContext } from "./thread-context"

export function TextPart({ part }: PartProps<typeof chatOptions, "text">) {
  const { sources } = useContext(ReplyContext)
  if (!part.content) return null
  return <MessageResponse sources={sources?.cited}>{part.content}</MessageResponse>
}
