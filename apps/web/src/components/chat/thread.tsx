import ArrowDown from "@gravity-ui/icons/ArrowDown"
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import ArrowUp from "@gravity-ui/icons/ArrowUp"
import Check from "@gravity-ui/icons/Check"
import Copy from "@gravity-ui/icons/Copy"
import { Button, TextArea, Tooltip } from "@heroui/react"
import type { UIMessage } from "@tanstack/ai"
import { useChat } from "@tanstack/ai-react"
import { TextPart } from "@tanstack/ai-react/ui"
import { useEffect, useLayoutEffect, useRef, useState } from "react"
import type { ReactNode } from "react"
import { v7 as uuidv7 } from "uuid"

import { apfelConnection } from "#/lib/llm/apfel-connection"

const SUGGESTIONS = [
  {
    title: "Summarize a topic",
    label: "like the history of the printing press",
    prompt: "Summarize the history of the printing press in five bullet points."
  },
  {
    title: "Draft a message",
    label: "to reschedule a meeting politely",
    prompt:
      "Write a short, polite message asking to move tomorrow's meeting to Thursday."
  },
  {
    title: "Explain a concept",
    label: "like how vaccines train the immune system",
    prompt: "Explain simply how vaccines train the immune system."
  },
  {
    title: "Brainstorm ideas",
    label: "for a weekend side project",
    prompt: "Give me five fun ideas for a weekend coding side project."
  }
]

function messageText(message: UIMessage) {
  return message.parts.map((p) => (p.type === "text" ? p.content : "")).join("")
}

/** Chat thread laid out like assistant-ui's default Thread, built from HeroUI components. */
export function Thread() {
  // One thread per page visit for now; saving and switching threads comes later.
  const [threadId] = useState(() => uuidv7())
  const { messages, sendMessage, isLoading, stop, reload, error } = useChat({
    connection: apfelConnection,
    threadId
  })

  const viewportRef = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    const el = viewportRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior })
  }

  // Follow the stream while the user is at the bottom; leave them alone if they scrolled up.
  useLayoutEffect(() => {
    if (atBottom) scrollToBottom("auto")
  }, [messages, isLoading, atBottom])

  const onScroll = () => {
    const el = viewportRef.current
    if (el) setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 32)
  }

  const last = messages.at(-1)
  const waitingForFirstToken =
    isLoading && (last?.role !== "assistant" || messageText(last) === "")

  return (
    <div
      ref={viewportRef}
      onScroll={onScroll}
      className="flex h-full flex-col items-center overflow-y-auto scroll-smooth px-4 pt-8"
    >
      {messages.length === 0 ? (
        <Welcome onPick={(prompt) => void sendMessage(prompt)} />
      ) : (
        <div className="flex w-full max-w-2xl flex-col gap-6 pb-4">
          {messages.map((message, index) =>
            message.role === "user" ? (
              <UserMessage key={message.id} text={messageText(message)} />
            ) : message.role === "assistant" ? (
              <AssistantMessage
                key={message.id}
                text={messageText(message)}
                isLast={index === messages.length - 1}
                isRunning={isLoading && index === messages.length - 1}
                onReload={() => void reload()}
              />
            ) : null
          )}
          {waitingForFirstToken && <TypingIndicator />}
          {error && !isLoading && (
            <div
              role="alert"
              className="rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger"
            >
              {error.message}
            </div>
          )}
        </div>
      )}

      <div className="sticky bottom-0 mt-auto flex w-full max-w-2xl flex-col items-center gap-3 bg-background pb-4">
        {!atBottom && (
          <Tooltip delay={300}>
            <Button
              isIconOnly
              aria-label="Scroll to bottom"
              size="sm"
              variant="secondary"
              className="absolute -top-12 rounded-full shadow-surface"
              onPress={() => scrollToBottom()}
            >
              <ArrowDown />
            </Button>
            <Tooltip.Content>Scroll to bottom</Tooltip.Content>
          </Tooltip>
        )}
        <Composer
          isRunning={isLoading}
          onSend={(text) => {
            setAtBottom(true)
            void sendMessage(text)
          }}
          onStop={stop}
        />
        <p className="text-xs text-muted">
          Apple Foundation Model · runs on your device · can make mistakes
        </p>
      </div>
    </div>
  )
}

function Welcome({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex w-full max-w-2xl flex-1 flex-col justify-center pb-8">
      <div className="mb-8 px-2">
        <h1 className="text-2xl font-semibold">Hello there!</h1>
        <p className="text-2xl text-muted">How can I help you today?</p>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <Button
            key={s.title}
            variant="outline"
            className="h-auto w-full flex-col items-start justify-start gap-0.5 rounded-3xl px-5 py-4 text-left"
            onPress={() => onPick(s.prompt)}
          >
            <span className="font-medium">{s.title}</span>
            <span className="text-sm font-normal text-muted">{s.label}</span>
          </Button>
        ))}
      </div>
    </div>
  )
}

function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] whitespace-pre-wrap rounded-3xl bg-surface-secondary px-5 py-2.5 break-words">
        {text}
      </div>
    </div>
  )
}

function AssistantMessage({
  text,
  isLast,
  isRunning,
  onReload
}: {
  text: string
  isLast: boolean
  isRunning: boolean
  onReload: () => void
}) {
  if (!text) return null
  return (
    <div className="group flex flex-col gap-1">
      <div className="markdown px-2 leading-7 break-words">
        {/* `role` here is the message sender for TextPart, not an ARIA role. */}
        {/* eslint-disable-next-line jsx-a11y/aria-role */}
        <TextPart content={text} role="assistant" />
      </div>
      {!isRunning && (
        <div
          className={`flex gap-1 px-1 transition-opacity ${isLast ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100"}`}
        >
          <CopyButton text={text} />
          {isLast && (
            <ActionButton label="Regenerate" onPress={onReload}>
              <ArrowRotateRight />
            </ActionButton>
          )}
        </div>
      )}
    </div>
  )
}

function ActionButton({
  label,
  onPress,
  children
}: {
  label: string
  onPress: () => void
  children: ReactNode
}) {
  return (
    <Tooltip delay={300}>
      <Button
        isIconOnly
        aria-label={label}
        size="sm"
        variant="ghost"
        onPress={onPress}
      >
        {children}
      </Button>
      <Tooltip.Content>{label}</Tooltip.Content>
    </Tooltip>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const t = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(t)
  }, [copied])
  return (
    <ActionButton
      label={copied ? "Copied" : "Copy"}
      onPress={() => {
        void navigator.clipboard.writeText(text)
        setCopied(true)
      }}
    >
      {copied ? <Check /> : <Copy />}
    </ActionButton>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-1 px-2 py-3" aria-label="Assistant is thinking">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="size-2 animate-bounce rounded-full bg-muted"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  )
}

function Composer({
  isRunning,
  onSend,
  onStop
}: {
  isRunning: boolean
  onSend: (text: string) => void
  onStop: () => void
}) {
  const [value, setValue] = useState("")
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Grow with the text, up to a limit.
  useLayoutEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  const submit = () => {
    const text = value.trim()
    if (!text || isRunning) return
    onSend(text)
    setValue("")
  }

  return (
    <form
      className="flex w-full items-end gap-2 rounded-3xl border border-border bg-surface px-2.5 py-2 shadow-surface focus-within:border-accent"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <TextArea
        ref={inputRef}
        aria-label="Message"
        placeholder="Write a message..."
        rows={1}
        value={value}
        autoFocus
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            submit()
          }
        }}
        className="max-h-[200px] min-h-10 flex-1 resize-none border-none bg-transparent px-2.5 py-2 shadow-none outline-none focus:ring-0"
      />
      {isRunning ? (
        <ActionButton label="Stop generating" onPress={onStop}>
          <span className="size-3 rounded-[3px] bg-current" />
        </ActionButton>
      ) : (
        <Tooltip delay={300}>
          <Button
            isIconOnly
            type="submit"
            aria-label="Send"
            size="sm"
            className="rounded-full"
            isDisabled={!value.trim()}
          >
            <ArrowUp />
          </Button>
          <Tooltip.Content>Send</Tooltip.Content>
        </Tooltip>
      )}
    </form>
  )
}
