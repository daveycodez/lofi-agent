/**
 * The `message` widget. A user message can be edited and sent again; an
 * assistant message shows its parts (thinking, tool calls, the answer)
 * through the kit's `Parts`, then its sources and actions. The layout
 * follows assistant-ui's thread (MIT, github.com/assistant-ui/assistant-ui).
 */
import ArrowDownToLine from "@gravity-ui/icons/ArrowDownToLine"
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import Check from "@gravity-ui/icons/Check"
import Copy from "@gravity-ui/icons/Copy"
import Ellipsis from "@gravity-ui/icons/Ellipsis"
import Pencil from "@gravity-ui/icons/Pencil"
import TrashBin from "@gravity-ui/icons/TrashBin"
import Volume from "@gravity-ui/icons/Volume"
import VolumeXmark from "@gravity-ui/icons/VolumeXmark"
import { Alert, Button, Dropdown, Label } from "@heroui/react"
import type { UIMessage } from "@tanstack/ai"
import type { MessageProps } from "@tanstack/ai-react/ui"
import { useContext, useEffect, useMemo, useState } from "react"

import { useSelectedModel } from "#/chat/model"
import { modelInfo, useModelStatus } from "#/chat/models"
import type { chatOptions } from "#/chat/options"
import { Action, Actions } from "#/components/ai/actions"
import { Message, MessageContent } from "#/components/ai/message"
import { ToolGroup } from "#/components/ai/tool"
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputToolbar
} from "#/components/ai/prompt-input"
import { SourceList } from "#/components/ai/sources"
import { TimeAgo } from "#/components/ai/time-ago"
import { messageText } from "#/lib/reply-sources"
import { ReplyContext, TranscriptContext, useThread } from "./thread-context"
import { ChatUI, useChatContext } from "./ui-components"

/**
 * The `message` widget. An assistant message lays out its parts itself
 * (see MessageParts), so the kit's `Parts` isn't used.
 */
export function ChatMessage({ message }: MessageProps<typeof chatOptions>) {
  if (message.role === "user") return <UserMessage message={message} />
  if (message.role === "assistant") {
    return <AssistantMessage message={message} />
  }
  return null
}

function UserMessage({ message }: { message: UIMessage }) {
  const { messages } = useChatContext()
  const { resend, remove } = useThread()
  const [isEditing, setIsEditing] = useState(false)
  const text = messageText(message)

  if (isEditing) {
    return (
      <EditMessage
        initial={text}
        onCancel={() => setIsEditing(false)}
        onUpdate={(next) => {
          setIsEditing(false)
          resend(
            messages.findIndex((m) => m.id === message.id),
            next
          )
        }}
        onDelete={() => {
          setIsEditing(false)
          remove(message.id)
        }}
      />
    )
  }

  return (
    <Message from="user">
      <MessageContent>{text}</MessageContent>
      <Actions>
        <CopyAction text={text} />
        <Action
          label="Regenerate"
          onPress={() =>
            resend(
              messages.findIndex((m) => m.id === message.id),
              text
            )
          }
        >
          <ArrowRotateRight />
        </Action>
        <Action label="Edit" onPress={() => setIsEditing(true)}>
          <Pencil />
        </Action>
      </Actions>
    </Message>
  )
}

/** A user message being edited, in a prompt input of its own. */
function EditMessage({
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
  return (
    <PromptInput label="Edit message" className="ms-auto w-full max-w-[85%]">
      <PromptInputTextarea
        value={value}
        maxHeight={320}
        autoFocus
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") onCancel()
        }}
      />
      <PromptInputToolbar>
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
      </PromptInputToolbar>
    </PromptInput>
  )
}

function AssistantMessage({ message }: { message: UIMessage }) {
  const { messages, isLoading, error, reload } = useChatContext()
  const { remove } = useThread()
  const sources = useContext(TranscriptContext).get(message.id)
  const index = messages.findIndex((m) => m.id === message.id)
  const isLast = index === messages.length - 1
  const isRunning = isLoading && isLast
  const text = messageText(message)
  const hasTools = message.parts.some((p) => p.type === "tool-call")
  const hasThinking = message.parts.some(
    (p) => p.type === "thinking" && p.content.trim()
  )
  const hasAnswer = !!text || hasTools
  // After a tool's result the model reads it before it writes again.
  const last = message.parts.at(-1)
  const afterTool =
    last?.type === "tool-result" ||
    (last?.type === "tool-call" && last.output !== undefined)
  const shownError = isLast && !isLoading ? error : undefined

  const reply = useMemo(
    () => ({ sources, isRunning, hasAnswer }),
    [sources, isRunning, hasAnswer]
  )

  return (
    <Message from="assistant">
      <ReplyContext.Provider value={reply}>
        <MessageParts message={message} />
      </ReplyContext.Provider>
      {isRunning && ((!hasAnswer && !hasThinking) || afterTool) && (
        <Working />
      )}
      {text && sources && sources.all.length > 0 && (
        <SourceList sources={sources.all} />
      )}
      {shownError && <ErrorAlert error={shownError} />}
      {!isRunning && (text || shownError) && (
        <ReplyActions
          text={text}
          canRegenerate={isLast}
          createdAt={message.createdAt}
          onReload={() => void reload()}
          onDelete={() => remove(message.id)}
        />
      )}
    </Message>
  )
}

type SelectedPart = Parameters<typeof ChatUI.Part>[0]["part"] & {
  key: string
}

/**
 * An assistant message's parts, as the kit renders them, with each run of
 * two or more tool calls gathered under one "N tool calls" accordion. A tool result
 * that belongs to a call shows in the call, as the kit's own `Parts` does.
 */
function MessageParts({ message }: { message: UIMessage }) {
  return (
    <ChatUI.Message message={message}>
      {(parts) => {
        const shown = (parts as Array<SelectedPart>).filter(
          (p) => !(p.key === "toolResult" && "matched" in p && p.matched)
        )
        const groups: Array<Array<SelectedPart>> = []
        for (const part of shown) {
          const last = groups.at(-1)
          if (part.key === "toolCall" && last?.[0]?.key === "toolCall") {
            last.push(part)
          } else {
            groups.push([part])
          }
        }
        return groups.map((group, i) =>
          group[0]?.key === "toolCall" && group.length > 1 ? (
            <ToolGroup key={i} count={group.length}>
              {group.map((part, j) => (
                <ChatUI.Part key={j} part={part} />
              ))}
            </ToolGroup>
          ) : (
            group.map((part, j) => <ChatUI.Part key={`${i}-${j}`} part={part} />)
          )
        )
      }}
    </ChatUI.Message>
  )
}

/**
 * The reply to the last user message before its first token arrives, or
 * after its run failed.
 */
export function PendingReply() {
  const { isLoading, error, reload } = useChatContext()
  return (
    <Message from="assistant">
      {isLoading && <Working />}
      {!isLoading && error && (
        <>
          <ErrorAlert error={error} />
          <ReplyActions text="" canRegenerate onReload={() => void reload()} />
        </>
      )}
    </Message>
  )
}

/** A size in bytes, as a browser's downloads list writes it: "412 MB", "1.10 GB". */
const formatBytes = (bytes: number) =>
  bytes >= 1e9
    ? `${(bytes / 1e9).toFixed(2)} GB`
    : `${Math.round(bytes / 1e6)} MB`

/**
 * What the model is doing until it writes, shimmering like "Thinking…": a
 * model downloads when the first message is sent to it (which model, and
 * how much of it is here), then loads into memory; after that it reads the
 * prompt, as it does on every turn and after every tool result.
 */
function Working() {
  const model = useSelectedModel()
  const status = useModelStatus(model)
  const loading = status.state === "loading" ? status : null
  const downloading =
    loading && !loading.cached && loading.progress < 1 ? loading : null
  const label = downloading
    ? `Downloading ${modelInfo(model).name}… ${
        downloading.totalBytes
          ? `${formatBytes(downloading.loadedBytes ?? 0)} of ${formatBytes(downloading.totalBytes)}`
          : `${Math.round(downloading.progress * 100)}%`
      }`
    : loading
      ? "Loading…"
      : "Processing…"
  // A row as tall as the Thoughts toggle, centered as its label is, so
  // nothing moves when the reply takes its place.
  return (
    <div className="flex min-h-8 w-fit items-center py-1 text-sm text-muted">
      <span className="text-shimmer">{label}</span>
    </div>
  )
}

function ErrorAlert({ error }: { error: Error }) {
  return (
    <Alert status="danger">
      <Alert.Indicator />
      <Alert.Content>
        <Alert.Description>{error.message}</Alert.Description>
      </Alert.Content>
    </Alert>
  )
}

function ReplyActions({
  text,
  canRegenerate,
  createdAt,
  onReload,
  onDelete
}: {
  text: string
  canRegenerate: boolean
  /** When the reply was written, shown as "N minutes ago". */
  createdAt?: Date | string
  onReload: () => void
  onDelete?: () => void
}) {
  return (
    <Actions>
      {text && <CopyAction text={text} />}
      {text && <ReadAloudAction text={text} />}
      {canRegenerate && (
        <Action label="Regenerate" onPress={onReload}>
          <ArrowRotateRight />
        </Action>
      )}
      {text && (
        <Dropdown>
          <Action label="More">
            <Ellipsis />
          </Action>
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
              <Dropdown.Item id="delete" textValue="Delete" variant="danger">
                <TrashBin className="size-4 text-danger" />
                <Label>Delete</Label>
              </Dropdown.Item>
            </Dropdown.Menu>
          </Dropdown.Popover>
        </Dropdown>
      )}
      {createdAt && <TimeAgo date={new Date(createdAt)} />}
    </Actions>
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
    <Action
      label={copied ? "Copied" : "Copy"}
      onPress={() => {
        void navigator.clipboard.writeText(text)
        setCopied(true)
      }}
    >
      {copied ? <Check /> : <Copy />}
    </Action>
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
    <Action
      label={speaking ? "Stop" : "Read"}
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
    </Action>
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
