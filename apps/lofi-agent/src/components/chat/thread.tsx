/**
 * One chat session: useChat creates its chat, and `ChatUI.Chat` (from
 * ./ui-components.tsx) renders it with this app's widgets. Around it go the
 * actions widgets use (send, edit and resend, delete) and the prompt
 * input's text.
 */
import { Spinner } from "@heroui/react"
import { useChat } from "@tanstack/ai-react"
import { useOptionalDbClient } from "@tanstack/react-db"
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"

import { useEffort, useSelectedModel } from "#/chat/model"
import type { ChatForwardedProps } from "#/chat/model"
import { setGenerating } from "#/chat/generating"
import { chatOptions } from "#/chat/options"
import { hasPendingInterrupts } from "#/chat/persistence"
import {
  checkMcpServers,
  useMcpApprovalTools,
  useMcpSettled
} from "#/lib/mcp"
import { touchThread } from "#/lib/threads"
import { ThreadSkeleton } from "./layout"
import { ComposerProvider, ThreadContext } from "./thread-context"
import type { ThreadActions } from "./thread-context"
import { ChatUI } from "./ui-components"

type ThreadProps = {
  threadId: string
  /** Called before each send (used to put the thread in the URL). */
  onSend: () => void
}

/**
 * The session, once the database is open: its messages and the chat's
 * saved state are there. Before that (and on the server) it's a skeleton.
 */
export function Thread({
  isNew,
  ...props
}: ThreadProps & {
  /** A session not saved yet, which opens on the welcome. */
  isNew: boolean
}) {
  const db = useOptionalDbClient()
  // Connect to the user's MCP servers so their tools are ready.
  useEffect(() => {
    if (db) checkMcpServers()
  }, [db])
  if (!db) return <ThreadSkeleton threadId={props.threadId} isNew={isNew} />
  return <ThreadGate {...props} />
}

function ThreadGate(props: ThreadProps) {
  const toolsKnown = useMcpSettled()
  // The chat client matches a saved approval to tool definitions once, as
  // it loads. A session saved waiting for one loads after the MCP tools
  // are known, so the approval can still be answered.
  const [waitForTools] = useState(() => hasPendingInterrupts(props.threadId))
  if (waitForTools && !toolsKnown) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner size="sm" color="current" className="text-muted" />
      </div>
    )
  }
  return <ThreadChat {...props} />
}

function ThreadChat({ threadId, onSend }: ThreadProps) {
  // Each run forwards the picked model and effort; the run picks its
  // adapter from them (src/chat/connection.ts).
  const model = useSelectedModel()
  const effort = useEffort()
  const forwardedProps: ChatForwardedProps = useMemo(
    () => ({ model, effort }),
    [model, effort]
  )
  // MCP tools that ask first: the chat client needs their definitions to
  // accept an approval (src/lib/mcp.ts).
  const tools = useMcpApprovalTools()
  const chat = useChat({ ...chatOptions, threadId, forwardedProps, tools })

  // The sidebar shimmers the title of a session that is replying.
  useEffect(() => {
    setGenerating(threadId, chat.isLoading)
    return () => setGenerating(threadId, false)
  }, [threadId, chat.isLoading])

  // Actions read the latest chat through refs, so they stay the same
  // functions and widgets that use them don't re-render on every chunk.
  const latest = useRef(chat)
  const onSendRef = useRef(onSend)
  useLayoutEffect(() => {
    latest.current = chat
    onSendRef.current = onSend
  })

  const actions: ThreadActions = useMemo(
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

  return (
    <ThreadContext.Provider value={actions}>
      <ComposerProvider>
        <ChatUI.Chat chat={chat} />
      </ComposerProvider>
    </ThreadContext.Provider>
  )
}
