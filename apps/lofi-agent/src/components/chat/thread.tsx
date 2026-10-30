/**
 * One chat session: `useAppChat` (from ./ui-components.tsx) creates its chat,
 * and `chat.AppChat` renders it with this app's widgets. Around it go the
 * actions widgets use (send, edit and resend, delete) and the prompt
 * input's text.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"

import { checkMcpServers } from "#/lib/mcp"
import { touchThread } from "#/lib/threads"
import { ComposerProvider, ThreadContext } from "./thread-context"
import type { ThreadActions } from "./thread-context"
import { useAppChat } from "./ui-components"

export function Thread({
  threadId,
  onSend
}: {
  threadId: string
  /** Called before each send (used to put the thread in the URL). */
  onSend: () => void
}) {
  const chat = useAppChat({ threadId })
  // Connect to the user's MCP servers so their tools are ready.
  useEffect(checkMcpServers, [])
  const timings = useReplyTimings(chat.isLoading, chat.messages)

  // Actions read the latest chat through refs, so they stay the same
  // functions and widgets that use them don't re-render on every chunk.
  const latest = useRef(chat)
  const onSendRef = useRef(onSend)
  useLayoutEffect(() => {
    latest.current = chat
    onSendRef.current = onSend
  })

  const actions = useMemo(
    () => ({
      threadId,
      send(text: string) {
        onSendRef.current()
        touchThread(threadId, text)
        void latest.current.sendMessage(text)
      },
      resend(index: number, text: string) {
        const current = latest.current
        if (current.isLoading) current.stop()
        onSendRef.current()
        touchThread(threadId, text)
        current.setMessages(current.messages.slice(0, index))
        void current.sendMessage(text, { whenBusy: "interrupt" })
      },
      remove(messageId: string) {
        const current = latest.current
        if (current.isLoading) current.stop()
        current.setMessages(current.messages.filter((m) => m.id !== messageId))
      }
    }),
    [threadId]
  )
  const value: ThreadActions = useMemo(
    () => ({ ...actions, timings }),
    [actions, timings]
  )

  return (
    <ThreadContext.Provider value={value}>
      <ComposerProvider>
        <chat.AppChat />
      </ComposerProvider>
    </ThreadContext.Provider>
  )
}

/** How long each reply took, by its last message id (this page load only). */
function useReplyTimings(
  isLoading: boolean,
  messages: ReadonlyArray<{ id: string; role: string }>
) {
  const [timings, setTimings] = useState<Record<string, number>>({})
  const runStart = useRef<number | null>(null)
  const wasLoading = useRef(false)
  useEffect(() => {
    if (isLoading && !wasLoading.current) runStart.current = performance.now()
    if (!isLoading && wasLoading.current && runStart.current !== null) {
      const last = messages.at(-1)
      if (last?.role === "assistant") {
        const seconds = (performance.now() - runStart.current) / 1000
        setTimings((t) => ({ ...t, [last.id]: seconds }))
      }
      runStart.current = null
    }
    wasLoading.current = isLoading
  }, [isLoading, messages])
  return timings
}
