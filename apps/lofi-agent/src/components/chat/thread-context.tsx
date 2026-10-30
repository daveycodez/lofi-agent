/**
 * This app's state around the chat kit: what widgets can do to the open
 * session, the prompt input's text, and what each reply's parts need to
 * know. The kit's own contexts are in src/chat/ui-context.ts.
 */
import { createContext, useContext, useState } from "react"
import type { ReactNode } from "react"

import type { ReplySources } from "#/lib/reply-sources"

/** The open session and what its widgets can do to it. */
export type ThreadActions = {
  threadId: string
  /** Sends a message, putting the session in the URL and the sidebar. */
  send: (text: string) => void
  /**
   * Replaces the chat from message `index` on with `text`, sent again. Works
   * mid-reply or mid-download: the current run stops first.
   */
  resend: (index: number, text: string) => void
  /** Removes one message, stopping the reply first if one is running. */
  remove: (messageId: string) => void
}

export const ThreadContext = createContext<ThreadActions | null>(null)

export function useThread() {
  const thread = useContext(ThreadContext)
  if (!thread) throw new Error("useThread must be used inside a Thread.")
  return thread
}

/**
 * The composer's text, shared by the composer and the empty-state
 * suggestions. A context of its own, so typing doesn't re-render messages.
 */
type ComposerState = { value: string; setValue: (value: string) => void }

const ComposerContext = createContext<ComposerState | null>(null)

export function ComposerProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState("")
  return (
    <ComposerContext.Provider value={{ value, setValue }}>
      {children}
    </ComposerContext.Provider>
  )
}

export function useComposer() {
  const composer = useContext(ComposerContext)
  if (!composer) throw new Error("useComposer must be used inside a Thread.")
  return composer
}

/** Sources of every reply in the transcript, by the reply's last message. */
export const TranscriptContext = createContext<Map<string, ReplySources>>(
  new Map()
)

/** What one assistant message's parts need to know about its reply. */
export type ReplyState = {
  /** The reply's sources, which `[n]` in its answer refers to. */
  sources?: ReplySources
  /** The reply is still running in this message. */
  isRunning: boolean
  /** This message has answer text or a tool call, after any thinking. */
  hasAnswer: boolean
}

export const ReplyContext = createContext<ReplyState>({
  isRunning: false,
  hasAnswer: false
})
