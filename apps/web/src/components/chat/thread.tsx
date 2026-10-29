/**
 * Chat thread. The layout follows assistant-ui's thread (MIT,
 * github.com/assistant-ui/assistant-ui); everything visual is HeroUI's
 * default theme and components. State comes from TanStack AI's useChat.
 */
import ArrowDown from "@gravity-ui/icons/ArrowDown"
import ArrowDownToLine from "@gravity-ui/icons/ArrowDownToLine"
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import ArrowUp from "@gravity-ui/icons/ArrowUp"
import Check from "@gravity-ui/icons/Check"
import Copy from "@gravity-ui/icons/Copy"
import Ellipsis from "@gravity-ui/icons/Ellipsis"
import Pencil from "@gravity-ui/icons/Pencil"
import StopFill from "@gravity-ui/icons/StopFill"
import Volume from "@gravity-ui/icons/Volume"
import VolumeXmark from "@gravity-ui/icons/VolumeXmark"
import {
  Alert,
  Button,
  Dropdown,
  InputGroup,
  Kbd,
  Label,
  Skeleton,
  Spinner,
  TextField,
  Tooltip
} from "@heroui/react"
import type { UIMessage } from "@tanstack/ai"
import { useChat } from "@tanstack/ai-react"
import { TextPart } from "@tanstack/ai-react/ui"
import { useEffect, useLayoutEffect, useRef, useState } from "react"
import type { ComponentProps, ReactNode } from "react"

import { ModelSelect } from "#/components/chat/model-select"
import { lfm3Status } from "#/lib/llm/lfm3-connection"
import { routedConnection, useCurrentModel } from "#/lib/llm/models"
import { chatPersistence, touchThread } from "#/lib/threads"

const SUGGESTIONS = [
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

function messageText(message: UIMessage) {
  return message.parts.map((p) => (p.type === "text" ? p.content : "")).join("")
}

/** Icon-only ghost button with a tooltip. */
function IconAction({
  label,
  children,
  ...props
}: Omit<ComponentProps<typeof Button>, "children" | "isIconOnly"> & {
  label: string
  children: ReactNode
}) {
  return (
    <Tooltip delay={300}>
      <Button
        isIconOnly
        size="sm"
        variant="ghost"
        aria-label={label}
        {...props}
      >
        {children}
      </Button>
      <Tooltip.Content>{label}</Tooltip.Content>
    </Tooltip>
  )
}

export function Thread({
  threadId,
  onSend
}: {
  threadId: string
  /** Called before each send (used to put the thread in the URL). */
  onSend: () => void
}) {
  const {
    messages,
    sendMessage,
    setMessages,
    isLoading,
    isHydrating,
    stop,
    reload,
    error
  } = useChat({
    connection: routedConnection,
    threadId,
    persistence: chatPersistence
  })
  const [composerValue, setComposerValue] = useState("")
  const isEmpty = messages.length === 0 && !isHydrating

  // Response time per assistant message (this session only).
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

  const viewportRef = useRef<HTMLDivElement>(null)
  const [atBottom, setAtBottom] = useState(true)
  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    const el = viewportRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior })
  }
  // Follow the stream while at the bottom; leave the user alone if they scrolled up.
  useLayoutEffect(() => {
    if (atBottom) scrollToBottom("auto")
  }, [messages, isLoading, atBottom])

  const send = (text: string) => {
    onSend()
    touchThread(threadId, text)
    setAtBottom(true)
    void sendMessage(text)
  }

  const lastIsUser = messages.at(-1)?.role === "user"

  const composer = (
    <Composer
      value={composerValue}
      onChange={setComposerValue}
      isRunning={isLoading}
      onSend={send}
      onStop={stop}
    />
  )

  // Empty thread: welcome, composer and suggestions centered together.
  if (isEmpty) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 pb-16">
        <div className="flex w-full max-w-2xl flex-col items-center">
          <h2 className="mb-8 text-center text-2xl font-semibold">
            How can I help you today?
          </h2>
          {composer}
          {!composerValue && (
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <Button
                  key={s.label}
                  size="sm"
                  variant="outline"
                  onPress={() => send(s.prompt)}
                >
                  {s.label}
                </Button>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div
      ref={viewportRef}
      onScroll={() => {
        const el = viewportRef.current
        if (el) {
          setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 32)
        }
      }}
      className="relative flex h-full flex-col overflow-y-auto px-6 pt-6"
    >
      <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col">
        {isHydrating ? (
          <ThreadHistorySkeleton />
        ) : (
          <div className="mb-14 flex flex-col gap-y-6">
            {messages.map((message, index) => {
              const isLast = index === messages.length - 1
              if (message.role === "user") {
                return (
                  <UserMessage
                    key={message.id}
                    text={messageText(message)}
                    isRunning={isLoading}
                    onEdit={(text) => {
                      setMessages(messages.slice(0, index))
                      send(text)
                    }}
                  />
                )
              }
              if (message.role === "assistant") {
                return (
                  <AssistantMessage
                    key={message.id}
                    text={messageText(message)}
                    isLast={isLast}
                    isRunning={isLoading && isLast}
                    seconds={timings[message.id]}
                    error={isLast && !isLoading ? error : undefined}
                    onReload={() => void reload()}
                  />
                )
              }
              return null
            })}
            {/* Placeholder reply while waiting for the first token, or for a failed run. */}
            {lastIsUser && (isLoading || error) && (
              <AssistantMessage
                text=""
                isLast
                isRunning={isLoading}
                error={isLoading ? undefined : error}
                onReload={() => void reload()}
              />
            )}
          </div>
        )}

        {/* Composer pinned to the bottom of the thread. */}
        <div className="sticky bottom-0 mt-auto flex flex-col bg-background pb-5">
          <div
            className={`absolute -top-12 self-center ${atBottom ? "invisible" : ""}`}
          >
            <IconAction
              label="Scroll to bottom"
              variant="outline"
              onPress={() => scrollToBottom()}
            >
              <ArrowDown />
            </IconAction>
          </div>
          {composer}
        </div>
      </div>
    </div>
  )
}

function ThreadHistorySkeleton() {
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

/** Grows a textarea with its content, up to `maxPx`, then scrolls. */
function useAutoGrow(value: string, maxPx: number) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, maxPx)}px`
    el.style.overflowY = el.scrollHeight > maxPx ? "auto" : "hidden"
  }, [value, maxPx])
  return ref
}

/** HeroUI's prompt input: TextField + InputGroup with a TextArea and a button row. */
function Composer({
  value,
  onChange,
  isRunning,
  onSend,
  onStop
}: {
  value: string
  onChange: (value: string) => void
  isRunning: boolean
  onSend: (text: string) => void
  onStop: () => void
}) {
  const textAreaRef = useAutoGrow(value, 192)
  // Can't send until a model is chosen and done downloading.
  const model = useCurrentModel()
  const lfm3 = lfm3Status.use()
  const isModelLoading =
    !model || (model === "lfm3" && lfm3.state === "loading")
  const canSend = !!value.trim() && !isRunning && !isModelLoading

  const submit = () => {
    if (!canSend) return
    onSend(value.trim())
    onChange("")
  }

  return (
    <TextField fullWidth aria-label="Message input" name="message">
      <InputGroup
        fullWidth
        className="flex flex-col gap-2 rounded-3xl pt-3.5 pb-2"
      >
        <InputGroup.TextArea
          ref={textAreaRef}
          placeholder="Send a message..."
          rows={1}
          value={value}
          autoFocus
          enterKeyHint="send"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (
              e.key === "Enter" &&
              !e.shiftKey &&
              !e.nativeEvent.isComposing
            ) {
              e.preventDefault()
              submit()
            }
          }}
          className="min-h-0 w-full flex-none resize-none px-3.5 py-0 text-base"
        />
        <InputGroup.Suffix className="flex w-full items-center gap-1.5 px-3 py-0">
          <ModelSelect />
          <div className="ms-auto flex items-center gap-1.5">
            {isRunning ? (
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  aria-label="Stop generating"
                  onPress={onStop}
                >
                  <StopFill />
                </Button>
                <Tooltip.Content>Stop generating</Tooltip.Content>
              </Tooltip>
            ) : (
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  aria-label="Send message"
                  isDisabled={!canSend}
                  onPress={submit}
                >
                  <ArrowUp />
                </Button>
                <Tooltip.Content className="flex items-center gap-1">
                  <p className="text-xs">Send</p>
                  <Kbd className="h-4 rounded-sm px-1">
                    <Kbd.Abbr keyValue="enter" />
                  </Kbd>
                </Tooltip.Content>
              </Tooltip>
            )}
          </div>
        </InputGroup.Suffix>
      </InputGroup>
    </TextField>
  )
}

function UserMessage({
  text,
  isRunning,
  onEdit
}: {
  text: string
  isRunning: boolean
  onEdit: (text: string) => void
}) {
  const [isEditing, setIsEditing] = useState(false)

  if (isEditing) {
    return (
      <EditComposer
        initial={text}
        onCancel={() => setIsEditing(false)}
        onUpdate={(next) => {
          setIsEditing(false)
          onEdit(next)
        }}
      />
    )
  }

  return (
    <div className="group flex items-center justify-end gap-2">
      {!isRunning && (
        <div className="opacity-0 group-hover:opacity-100 focus-within:opacity-100">
          <IconAction label="Edit message" onPress={() => setIsEditing(true)}>
            <Pencil />
          </IconAction>
        </div>
      )}
      <div className="max-w-[80%] rounded-2xl bg-default px-4 py-2 wrap-break-word whitespace-pre-wrap text-default-foreground">
        {text}
      </div>
    </div>
  )
}

function EditComposer({
  initial,
  onCancel,
  onUpdate
}: {
  initial: string
  onCancel: () => void
  onUpdate: (text: string) => void
}) {
  const [value, setValue] = useState(initial)
  const textAreaRef = useAutoGrow(value, 320)
  return (
    <TextField
      fullWidth
      aria-label="Edit message"
      className="ms-auto w-full max-w-[85%]"
    >
      <InputGroup
        fullWidth
        className="flex flex-col gap-2 rounded-3xl pt-3.5 pb-2"
      >
        <InputGroup.TextArea
          ref={textAreaRef}
          value={value}
          rows={1}
          autoFocus
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onCancel()
          }}
          className="min-h-0 w-full flex-none resize-none px-3.5 py-0 text-base"
        />
        <InputGroup.Suffix className="flex w-full items-center justify-end gap-1.5 px-3 py-0">
          <Button size="sm" variant="ghost" onPress={onCancel}>
            Cancel
          </Button>
          <Button
            size="sm"
            isDisabled={!value.trim()}
            onPress={() => onUpdate(value.trim())}
          >
            Update
          </Button>
        </InputGroup.Suffix>
      </InputGroup>
    </TextField>
  )
}

function AssistantMessage({
  text,
  isLast,
  isRunning,
  seconds,
  error,
  onReload
}: {
  text: string
  isLast: boolean
  isRunning: boolean
  seconds?: number
  error?: Error
  onReload: () => void
}) {
  return (
    <div className="group flex flex-col gap-2">
      {text ? (
        <div className="markdown leading-relaxed wrap-break-word">
          {/* `role` is TextPart's message-sender prop, not an ARIA role. */}
          {/* eslint-disable-next-line jsx-a11y/aria-role */}
          <TextPart content={text} role="assistant" />
        </div>
      ) : (
        isRunning && (
          <div className="flex items-center gap-2 text-sm text-muted">
            <Spinner size="sm" color="current" />
            Thinking…
          </div>
        )
      )}
      {error && (
        <Alert status="danger">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Description>{error.message}</Alert.Description>
          </Alert.Content>
        </Alert>
      )}
      {!isRunning && (text || error) && (
        // Hidden until hover, except on the last reply.
        <div
          className={`flex items-center gap-1 ${isLast ? "" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100"}`}
        >
          {text && <CopyAction text={text} />}
          {text && <ReadAloudAction text={text} />}
          {isLast && (
            <IconAction label="Regenerate response" onPress={onReload}>
              <ArrowRotateRight />
            </IconAction>
          )}
          {text && (
            <Dropdown>
              <IconAction label="More actions">
                <Ellipsis />
              </IconAction>
              <Dropdown.Popover placement="bottom start">
                <Dropdown.Menu
                  onAction={(key) => {
                    if (key === "export") downloadMarkdown(text)
                  }}
                >
                  <Dropdown.Item id="export" textValue="Export as Markdown">
                    <ArrowDownToLine className="size-4" />
                    <Label>Export as Markdown</Label>
                  </Dropdown.Item>
                </Dropdown.Menu>
              </Dropdown.Popover>
            </Dropdown>
          )}
          {seconds !== undefined && (
            <span className="px-1 text-xs text-muted tabular-nums">
              {seconds.toFixed(2)}s
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function CopyAction({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const t = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(t)
  }, [copied])
  return (
    <IconAction
      label={copied ? "Copied" : "Copy response"}
      onPress={() => {
        void navigator.clipboard.writeText(text)
        setCopied(true)
      }}
    >
      {copied ? <Check /> : <Copy />}
    </IconAction>
  )
}

/** Reads the reply with the browser's built-in, on-device speech. */
function ReadAloudAction({ text }: { text: string }) {
  const [speaking, setSpeaking] = useState(false)
  useEffect(() => () => window.speechSynthesis?.cancel(), [])
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return null
  }
  return (
    <IconAction
      label={speaking ? "Stop reading" : "Read aloud"}
      onPress={() => {
        window.speechSynthesis.cancel()
        if (speaking) {
          setSpeaking(false)
          return
        }
        const utterance = new SpeechSynthesisUtterance(
          text.replace(/[*_`#>]/g, "")
        )
        utterance.onend = () => setSpeaking(false)
        utterance.onerror = () => setSpeaking(false)
        setSpeaking(true)
        window.speechSynthesis.speak(utterance)
      }}
    >
      {speaking ? <VolumeXmark /> : <Volume />}
    </IconAction>
  )
}

function downloadMarkdown(text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }))
  const a = document.createElement("a")
  a.href = url
  a.download = "message.md"
  a.click()
  URL.revokeObjectURL(url)
}
