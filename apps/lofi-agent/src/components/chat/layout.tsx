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
import { Suggestion, Suggestions } from "#/components/ai/suggestions"
import { sourcesByReply } from "#/lib/reply-sources"
import { PendingReply } from "./message"
import { ReplyStatsLine } from "./reply-stats"
import { TranscriptContext, useComposer, useThread } from "./thread-context"
import { useChatContext } from "./ui-components"

const PROMPTS = [
  {
    label: "Summarize a topic",
    prompt: "Summarize the history of the printing press in five bullet points."
  },
  {
    label: "Draft a message",
    prompt:
      "Write a short, polite message asking to move tomorrow's meeting to Thursday."
  },
  {
    label: "Explain a concept",
    prompt: "Explain simply how vaccines train the immune system."
  },
  {
    label: "Brainstorm ideas",
    prompt: "Give me five fun ideas for a weekend coding side project."
  },
  { label: "Write a haiku", prompt: "Write a haiku about autumn." },
  {
    label: "Plan a dinner party",
    prompt: "Help me plan a simple dinner party for six people."
  }
]

export function ChatLayout({
  Messages,
  Interrupts,
  Queue,
  Input
}: LayoutProps<typeof chatOptions>) {
  const { messages, isHydrating, queue } = useChatContext()
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
        <ConversationScrollButton />
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
        className="mb-6 flex flex-col gap-y-4"
        onPointerDownCapture={(e) => releaseOnToggle(e.target, stopScroll)}
        onKeyDownCapture={(e) => releaseOnToggle(e.target, stopScroll)}
      >
        <Messages />
        {/* The reply before its first token, or a run that failed. */}
        {last?.role === "user" && (isLoading || error) && <PendingReply />}
        {/* Approvals and other pauses waiting for an answer. */}
        <Interrupts />
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
  const { value } = useComposer()
  const { send } = useThread()
  return (
    <ConversationEmptyState title="How can I help you today?">
      <Input />
      {!value && (
        <Suggestions>
          {PROMPTS.map((p) => (
            <Suggestion key={p.label} onPress={() => send(p.prompt)}>
              {p.label}
            </Suggestion>
          ))}
        </Suggestions>
      )}
    </ConversationEmptyState>
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
