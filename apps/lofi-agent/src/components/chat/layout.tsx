/**
 * The `layout` widget: the conversation, which follows the reply while
 * you're at the bottom, and the prompt input under it. An empty session
 * centers a welcome, the prompt input and suggestions.
 */
import { Skeleton } from "@heroui/react"
import type { LayoutProps } from "@tanstack/ai-react/ui"
import { useEffect, useMemo } from "react"
import type { ComponentType } from "react"
import { useStickToBottomContext } from "use-stick-to-bottom"

import type { chatOptions } from "#/chat/options"
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton
} from "#/components/ai/conversation"
import { sourcesByReply } from "#/lib/reply-sources"
import { PromptInputSkeleton } from "./input"
import { PendingReply } from "./message"
import { ReplyStatsLine } from "./reply-stats"
import { TranscriptContext, useThread } from "./thread-context"
import { useChatContext } from "./ui-components"

export function ChatLayout({
  Messages,
  Interrupts,
  Queue,
  Input
}: LayoutProps<typeof chatOptions>) {
  const { messages, isHydrating, isLoading, queue } = useChatContext()
  const { threadId } = useThread()

  if (messages.length === 0 && !isHydrating) {
    return <EmptyState Input={Input} />
  }

  return (
    <Conversation>
      <ConversationContent>
        {isHydrating ? (
          <TranscriptSkeleton />
        ) : (
          <Transcript Messages={Messages} Interrupts={Interrupts} />
        )}
      </ConversationContent>
      <div className="relative mx-auto flex w-full max-w-[calc(714px+3rem)] flex-col px-6 pb-2">
        <ConversationScrollButton isLoading={isLoading} />
        {queue.length > 0 && (
          <div className="mb-2 flex flex-col gap-1">
            <p className="px-1 text-xs font-medium text-muted">Queued</p>
            <Queue />
          </div>
        )}
        <Input />
        {/* Token counts and speed of the latest reply. */}
        <ReplyStatsLine threadId={threadId} />
      </div>
    </Conversation>
  )
}

function Transcript({
  Messages,
  Interrupts
}: {
  Messages: ComponentType
  Interrupts: ComponentType
}) {
  const { messages, isLoading, error } = useChatContext()
  const sources = useMemo(() => sourcesByReply(messages), [messages])
  const last = messages.at(-1)

  // Sending a message jumps back to the bottom, to follow the reply.
  const { scrollToBottom, stopScroll } = useStickToBottomContext()
  const lastUserId = messages.filter((m) => m.role === "user").at(-1)?.id
  useEffect(() => {
    if (lastUserId) void scrollToBottom()
  }, [lastUserId, scrollToBottom])

  return (
    <TranscriptContext.Provider value={sources}>
      {/* Opening or closing something (thoughts, a tool call, sources)
          stops following the bottom first, so what was opened stays put
          rather than being scrolled up as the transcript grows. */}
      <div
        className="mb-6 flex flex-col gap-y-8"
        onPointerDownCapture={(e) => releaseOnToggle(e.target, stopScroll)}
        onKeyDownCapture={(e) => releaseOnToggle(e.target, stopScroll)}
      >
        <Messages />
        {/* The reply before its first token, or a run that failed. */}
        {last?.role === "user" && (isLoading || error) && <PendingReply />}
        {/* Approvals and other pauses waiting for an answer, 8px under the
            reply they belong to, as a part of it would be. */}
        <div className="-mt-6 flex flex-col gap-2 empty:hidden">
          <Interrupts />
        </div>
      </div>
    </TranscriptContext.Provider>
  )
}

/** Stops following the bottom when `target` is inside a toggle. */
function releaseOnToggle(target: EventTarget, stopScroll: () => void) {
  if (target instanceof Element && target.closest("[aria-expanded]")) {
    stopScroll()
  }
}

function EmptyState({ Input }: { Input: ComponentType }) {
  return (
    <ConversationEmptyState title="How can I help you today?">
      <Input />
    </ConversationEmptyState>
  )
}

/**
 * A session before the database opens (and on the server): a new session's
 * welcome, or a saved one's transcript as a skeleton, with the prompt input.
 * Laid out like ChatLayout, so nothing moves when the chat takes over.
 */
export function ThreadSkeleton({
  threadId,
  isNew
}: {
  threadId: string
  isNew: boolean
}) {
  if (isNew) {
    return (
      <ConversationEmptyState title="How can I help you today?">
        <PromptInputSkeleton />
      </ConversationEmptyState>
    )
  }
  return (
    <Conversation>
      <ConversationContent>
        <TranscriptSkeleton />
      </ConversationContent>
      <div className="relative mx-auto flex w-full max-w-[calc(714px+3rem)] flex-col px-6 pb-2">
        <PromptInputSkeleton />
        <ReplyStatsLine threadId={threadId} />
      </div>
    </Conversation>
  )
}

function TranscriptSkeleton() {
  return (
    <div role="status" className="flex flex-col gap-y-6">
      <span className="sr-only">Loading conversation</span>
      <Skeleton className="ml-auto h-9 w-2/5 rounded-2xl" />
      <div className="flex flex-col gap-y-2">
        <Skeleton className="h-4 w-11/12 rounded-md" />
        <Skeleton className="h-4 w-4/5 rounded-md" />
        <Skeleton className="h-4 w-3/5 rounded-md" />
      </div>
    </div>
  )
}
