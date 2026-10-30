import Clock from "@gravity-ui/icons/Clock"
import Xmark from "@gravity-ui/icons/Xmark"
import type { QueueProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"
import { Action } from "#/components/ai/actions"

function queuedLabel(
  content: QueueProps<typeof chatOptions>["item"]["content"]
) {
  if (typeof content === "string") return content
  if (typeof content.content === "string") return content.content
  const text = content.content.find((part) => part.type === "text")
  if (text?.type === "text") return text.content
  return "Queued message"
}

/** A message sent while a reply runs, waiting its turn. */
export function ChatQueueItem({ item }: QueueProps<typeof chatOptions>) {
  return (
    <div className="flex items-center gap-2 rounded-xl bg-surface-secondary py-1 ps-3 pe-1 text-xs">
      <Clock className="size-3 shrink-0 text-muted" />
      <span className="min-w-0 flex-1 truncate text-muted">
        {queuedLabel(item.content)}
      </span>
      <Action
        label="Cancel queued message"
        onPress={() => item.cancelQueued()}
      >
        <Xmark />
      </Action>
    </div>
  )
}
