/**
 * Chat thread. The layout follows assistant-ui's thread (MIT,
 * github.com/assistant-ui/assistant-ui); everything visual is HeroUI's
 * default theme and components. State comes from TanStack AI's useChat.
 */
import ArrowDown from "@gravity-ui/icons/ArrowDown"
import ArrowDownToLine from "@gravity-ui/icons/ArrowDownToLine"
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import ArrowUp from "@gravity-ui/icons/ArrowUp"
import Bulb from "@gravity-ui/icons/Bulb"
import Check from "@gravity-ui/icons/Check"
import Copy from "@gravity-ui/icons/Copy"
import Ellipsis from "@gravity-ui/icons/Ellipsis"
import Microphone from "@gravity-ui/icons/Microphone"
import Pencil from "@gravity-ui/icons/Pencil"
import StopFill from "@gravity-ui/icons/StopFill"
import TrashBin from "@gravity-ui/icons/TrashBin"
import TriangleExclamation from "@gravity-ui/icons/TriangleExclamation"
import Volume from "@gravity-ui/icons/Volume"
import VolumeXmark from "@gravity-ui/icons/VolumeXmark"
import Wrench from "@gravity-ui/icons/Wrench"
import {
  Alert,
  Button,
  Disclosure,
  Dropdown,
  InputGroup,
  Kbd,
  Label,
  Skeleton,
  Spinner,
  TextField,
  ToggleButton,
  Tooltip
} from "@heroui/react"
import type { UIMessage } from "@tanstack/ai"
import { useChat } from "@tanstack/ai-react"
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import type { ComponentProps, ReactNode } from "react"

import { Markdown } from "#/components/chat/markdown"
import { ModelSelect } from "#/components/chat/model-select"
import { SourceList } from "#/components/chat/sources"
import { useDictation } from "#/lib/dictation"
import { checkMcpServers, useMcpTools } from "#/lib/mcp"
import { parseSources } from "#/lib/sources"
import { useWebSearchTools } from "#/lib/web-search"
import type { Source } from "#/lib/sources"
import { splitThinking } from "#/lib/thinking"
import {
  canThink,
  routedConnection,
  thinkingStore,
  useCurrentModel,
  useModelStatus
} from "#/lib/llm/models"
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

/**
 * A reply's thinking (its thinking parts) and its answer (its text). Replies
 * saved by earlier versions kept the thinking as <think> tags in the text.
 */
function replyOf(message: UIMessage) {
  const { thinking, answer } = splitThinking(messageText(message))
  const thoughts = message.parts
    .map((p) => (p.type === "thinking" ? p.content.trim() : ""))
    .filter(Boolean)
  const toolCalls = message.parts.flatMap((p): Array<ToolCallView> => {
    if (p.type !== "tool-call") return []
    const result = message.parts.find(
      (r) => r.type === "tool-result" && r.toolCallId === p.id
    )
    const error =
      result?.type === "tool-result" &&
      (result.error || result.state === "error")
        ? (result.error ?? String(result.content))
        : p.state === "error"
          ? "The tool failed."
          : undefined
    const output =
      p.output !== undefined
        ? typeof p.output === "string"
          ? p.output
          : JSON.stringify(p.output, null, 2)
        : result?.type === "tool-result" && typeof result.content === "string"
          ? result.content
          : undefined
    return [{ id: p.id, name: p.name, arguments: p.arguments, output, error }]
  })
  return {
    thinking: [...thoughts, thinking].filter(Boolean).join("\n\n"),
    text: answer,
    toolCalls
  }
}

/** The sources in each tool result of a message, in order. */
function sourceListsOf(message: UIMessage) {
  return message.parts
    .map((p) =>
      p.type === "tool-call" && typeof p.output === "string"
        ? parseSources(p.output)
        : p.type === "tool-result" && typeof p.content === "string"
          ? parseSources(p.content)
          : []
    )
    .filter((list) => list.length > 0)
}

type ReplySources = {
  /** Every source the reply's tools returned, once each. */
  all: Array<Source>
  /** The latest tool result's sources, which `[n]` in the answer refers to. */
  cited: Array<Source>
}

/**
 * Sources for each reply, keyed by the reply's last message. A reply that
 * called tools spans several assistant messages: the calls, then the answer.
 */
function sourcesByReply(messages: ReadonlyArray<UIMessage>) {
  const byReply = new Map<string, ReplySources>()
  let lists: Array<Array<Source>> = []
  messages.forEach((message, i) => {
    if (message.role !== "assistant") {
      lists = []
      return
    }
    lists.push(...sourceListsOf(message))
    if (messages[i + 1]?.role === "assistant" || lists.length === 0) return
    const seen = new Set<string>()
    byReply.set(message.id, {
      all: lists.flat().filter((s) => !seen.has(s.url) && seen.add(s.url)),
      cited: lists.at(-1) ?? []
    })
  })
  return byReply
}

type ToolCallView = {
  id: string
  name: string
  /** JSON arguments, as the model wrote them. */
  arguments: string
  output?: string
  error?: string
}

/** Pretty-prints JSON arguments, or returns them as they are. */
function prettyArguments(json: string) {
  try {
    return JSON.stringify(JSON.parse(json), null, 2)
  } catch {
    return json
  }
}

/** A tool the model called: its name, and its arguments and result inside. */
function ToolCallBlock({
  call,
  isRunning
}: {
  call: ToolCallView
  isRunning: boolean
}) {
  const pending = call.output === undefined && !call.error
  return (
    <Disclosure>
      <Disclosure.Heading>
        <Button
          slot="trigger"
          size="sm"
          variant="ghost"
          className="-ms-3 text-muted"
        >
          {pending && isRunning ? (
            <Spinner size="sm" color="current" />
          ) : call.error ? (
            <TriangleExclamation className="text-danger" />
          ) : (
            <Wrench />
          )}
          <span className="font-mono text-xs">{call.name}</span>
          {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
          <Disclosure.Indicator className="ms-0" />
        </Button>
      </Disclosure.Heading>
      <Disclosure.Content>
        <Disclosure.Body className="flex flex-col gap-2 border-s-2 border-separator ps-3 text-xs">
          <p className="text-muted">Arguments</p>
          <pre className="max-h-48 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap">
            {prettyArguments(call.arguments)}
          </pre>
          {(call.output !== undefined || call.error) && (
            <>
              <p className="text-muted">{call.error ? "Error" : "Result"}</p>
              <pre
                className={`max-h-60 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap ${call.error ? "text-danger" : ""}`}
              >
                {call.error ?? call.output}
              </pre>
            </>
          )}
        </Disclosure.Body>
      </Disclosure.Content>
    </Disclosure>
  )
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
        className="text-muted"
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
  // Built-in web search first: small models tend to pick the first tool that
  // might fit. useChat runs these tools in the browser when called.
  const webSearchTools = useWebSearchTools()
  const mcpTools = useMcpTools()
  const tools = useMemo(() => {
    const names = new Set<string>(webSearchTools.map((t) => t.name))
    return [...webSearchTools, ...mcpTools.filter((t) => !names.has(t.name))]
  }, [webSearchTools, mcpTools])
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
    persistence: chatPersistence,
    tools
  })
  // Connect to the user's MCP servers so their tools are ready.
  useEffect(checkMcpServers, [])
  const [composerValue, setComposerValue] = useState("")
  const isEmpty = messages.length === 0 && !isHydrating
  // A model downloads when the first message is sent to it; the reply's
  // "Thinking…" shows the download's progress until it's done.
  const modelStatus = useModelStatus(useCurrentModel())
  const downloading =
    modelStatus.state === "loading" &&
    !modelStatus.cached &&
    modelStatus.progress < 1
      ? modelStatus.progress
      : null
  const replySources = useMemo(() => sourcesByReply(messages), [messages])

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
                    onDelete={() =>
                      setMessages(messages.filter((m) => m.id !== message.id))
                    }
                  />
                )
              }
              if (message.role === "assistant") {
                return (
                  <AssistantMessage
                    key={message.id}
                    {...replyOf(message)}
                    sources={replySources.get(message.id)}
                    // After a tool call, the reply goes on in a new message.
                    continues={messages[index - 1]?.role === "assistant"}
                    isLast={isLast}
                    isRunning={isLoading && isLast}
                    downloading={isLast ? downloading : null}
                    seconds={timings[message.id]}
                    error={isLast && !isLoading ? error : undefined}
                    onReload={() => void reload()}
                    onDelete={() =>
                      setMessages(messages.filter((m) => m.id !== message.id))
                    }
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
                downloading={downloading}
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
              className=""
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
  const model = useCurrentModel()
  const canSend = !!value.trim() && !isRunning

  // Dictation adds to whatever was already typed when it started.
  const dictation = useDictation()
  const toggleDictation = () => {
    if (dictation.isListening) return dictation.stop()
    const before = value.trimEnd()
    dictation.start((text) =>
      onChange(before ? `${before} ${text.trimStart()}` : text.trimStart())
    )
  }

  const submit = () => {
    if (!canSend) return
    dictation.cancel()
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
          {canThink(model) && <ThinkToggle />}
          <div className="ms-auto flex items-center gap-1">
            {dictation.isSupported && (
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  size="sm"
                  variant="ghost"
                  aria-label={
                    dictation.isListening
                      ? "Stop voice input"
                      : "Start voice input"
                  }
                  aria-pressed={dictation.isListening}
                  onPress={toggleDictation}
                  className={
                    dictation.isListening ? "text-danger" : "text-muted"
                  }
                >
                  <Microphone />
                </Button>
                <Tooltip.Content>
                  {dictation.isListening
                    ? "Stop voice input"
                    : "Start voice input"}
                </Tooltip.Content>
              </Tooltip>
            )}
            {isRunning ? (
              <Tooltip delay={300}>
                <Button
                  isIconOnly
                  size="sm"
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
                  size="sm"
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
  onEdit,
  onDelete
}: {
  text: string
  isRunning: boolean
  onEdit: (text: string) => void
  onDelete: () => void
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
        onDelete={() => {
          setIsEditing(false)
          onDelete()
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
  onUpdate,
  onDelete
}: {
  initial: string
  onCancel: () => void
  onUpdate: (text: string) => void
  onDelete: () => void
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
        <InputGroup.Suffix className="flex w-full items-center gap-1.5 px-3 py-0">
          <Button size="sm" variant="danger" onPress={onDelete}>
            Delete
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onPress={onCancel}
            className="ms-auto"
          >
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

/** Turns on reasoning before answering, for models that can think. */
function ThinkToggle() {
  const isThinking = thinkingStore.use()
  return (
    <Tooltip delay={300}>
      <ToggleButton
        size="sm"
        variant="ghost"
        isSelected={isThinking}
        onChange={(on) => {
          thinkingStore.set(on)
        }}
        aria-label="Think before answering"
        className={isThinking ? undefined : "text-muted"}
      >
        <Bulb />
        Think
      </ToggleButton>
      <Tooltip.Content>
        {isThinking
          ? "Thinking on: slower, more careful"
          : "Think before answering"}
      </Tooltip.Content>
    </Tooltip>
  )
}

/** The model's thinking, collapsed under a "Thinking" toggle. */
function ThinkingBlock({
  thinking,
  isThinking
}: {
  thinking: string
  isThinking: boolean
}) {
  return (
    <Disclosure>
      <Disclosure.Heading>
        <Button
          slot="trigger"
          size="sm"
          variant="ghost"
          className="-ms-3 text-muted"
        >
          {isThinking && <Spinner size="sm" color="current" />}
          {isThinking ? "Thinking…" : "Thoughts"}
          {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
          <Disclosure.Indicator className="ms-0" />
        </Button>
      </Disclosure.Heading>
      <Disclosure.Content>
        <Disclosure.Body className="markdown prose prose-sm max-w-none border-s-2 border-separator ps-3 text-muted wrap-break-word">
          <Markdown>{thinking}</Markdown>
        </Disclosure.Body>
      </Disclosure.Content>
    </Disclosure>
  )
}

function AssistantMessage({
  text,
  thinking = "",
  toolCalls = [],
  sources,
  continues = false,
  isLast,
  isRunning,
  downloading = null,
  seconds,
  error,
  onReload,
  onDelete
}: {
  text: string
  thinking?: string
  toolCalls?: Array<ToolCallView>
  /** Sources from the reply's tool calls, shown under the answer. */
  sources?: ReplySources
  /** Continues the assistant message before it, as part of the same reply. */
  continues?: boolean
  isLast: boolean
  isRunning: boolean
  /** How much of the model has downloaded (0-1), while it downloads. */
  downloading?: number | null
  seconds?: number
  error?: Error
  onReload: () => void
  onDelete?: () => void
}) {
  return (
    // A continued reply sits as close as the parts within a message
    // (gap-2), not the gap between messages (gap-y-6).
    <div className={`group flex flex-col gap-2 ${continues ? "-mt-4" : ""}`}>
      {thinking && (
        <ThinkingBlock
          thinking={thinking}
          isThinking={isRunning && !text && toolCalls.length === 0}
        />
      )}
      {toolCalls.map((call) => (
        <ToolCallBlock key={call.id} call={call} isRunning={isRunning} />
      ))}
      {text ? (
        <div className="markdown prose max-w-none wrap-break-word">
          <Markdown sources={sources?.cited}>{text}</Markdown>
        </div>
      ) : (
        isRunning &&
        !thinking &&
        toolCalls.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-muted">
            <Spinner size="sm" color="current" />
            {downloading === null
              ? "Thinking…"
              : `Downloading ${Math.round(downloading * 100)}%`}
          </div>
        )
      )}
      {text && sources && sources.all.length > 0 && (
        <SourceList sources={sources.all} />
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
                    if (key === "delete") onDelete?.()
                  }}
                >
                  <Dropdown.Item id="export" textValue="Export as Markdown">
                    <ArrowDownToLine className="size-4" />
                    <Label>Export as Markdown</Label>
                  </Dropdown.Item>
                  <Dropdown.Item
                    id="delete"
                    textValue="Delete"
                    variant="danger"
                  >
                    <TrashBin className="size-4 text-danger" />
                    <Label>Delete</Label>
                  </Dropdown.Item>
                </Dropdown.Menu>
              </Dropdown.Popover>
            </Dropdown>
          )}
          {seconds !== undefined && (
            <span className="px-1 text-xs text-muted tabular-nums">
              {Number(seconds.toFixed(1))}s
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
