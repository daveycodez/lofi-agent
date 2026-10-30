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
import { Alert, Button, Dropdown, Label, Spinner } from "@heroui/react"
import type { UIMessage } from "@tanstack/ai"
import type { MessageProps } from "@tanstack/ai-react/ui"
import { useContext, useEffect, useMemo, useState } from "react"
import type { ComponentType } from "react"

import { useCurrentModel, useModelStatus } from "#/chat/models"
import type { chatOptions } from "#/chat/options"
import { Action, Actions } from "#/components/ai/actions"
import { Message, MessageContent } from "#/components/ai/message"
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputToolbar
} from "#/components/ai/prompt-input"
import { SourceList } from "#/components/ai/sources"
import { messageText } from "#/lib/reply-sources"
import { ReplyContext, TranscriptContext, useThread } from "./thread-context"
import { useChatContext } from "./ui-components"

export function ChatMessage({
  message,
  Parts
}: MessageProps<typeof chatOptions>) {
  if (message.role === "user") return <UserMessage message={message} />
  if (message.role === "assistant") {
    return <AssistantMessage message={message} Parts={Parts} />
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
      <div>
        <Action label="Edit message" onPress={() => setIsEditing(true)}>
          <Pencil />
        </Action>
      </div>
      <MessageContent>{text}</MessageContent>
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

function AssistantMessage({
  message,
  Parts
}: {
  message: UIMessage
  Parts: ComponentType
}) {
  const { messages, isLoading, error, reload } = useChatContext()
  const { remove, timings } = useThread()
  const sources = useContext(TranscriptContext).get(message.id)
  const index = messages.findIndex((m) => m.id === message.id)
  const isLast = index === messages.length - 1
  const isRunning = isLoading && isLast
  // After a tool call, the reply goes on in a new message.
  const continues = messages[index - 1]?.role === "assistant"
  const text = messageText(message)
  const hasTools = message.parts.some((p) => p.type === "tool-call")
  const hasThinking = message.parts.some(
    (p) => p.type === "thinking" && p.content.trim()
  )
  const hasAnswer = !!text || hasTools
  const shownError = isLast && !isLoading ? error : undefined

  const reply = useMemo(
    () => ({ sources, isRunning, hasAnswer }),
    [sources, isRunning, hasAnswer]
  )

  return (
    // A continued reply sits as close as the parts within a message
    // (gap-2), not the gap between messages (gap-y-6).
    <Message from="assistant" className={continues ? "-mt-4" : ""}>
      <ReplyContext.Provider value={reply}>
        <Parts />
      </ReplyContext.Provider>
      {isRunning && !hasAnswer && !hasThinking && <Working />}
      {text && sources && sources.all.length > 0 && (
        <SourceList sources={sources.all} />
      )}
      {shownError && <ErrorAlert error={shownError} />}
      {!isRunning && (text || shownError) && (
        <ReplyActions
          text={text}
          canRegenerate={isLast}
          seconds={timings[message.id]}
          onReload={() => void reload()}
          onDelete={() => remove(message.id)}
        />
      )}
    </Message>
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

/**
 * "Thinking…" until the reply starts. A model downloads when the first
 * message is sent to it; its progress shows here until it's done.
 */
function Working() {
  const status = useModelStatus(useCurrentModel())
  const downloading =
    status.state === "loading" && !status.cached && status.progress < 1
      ? status.progress
      : null
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <Spinner size="sm" color="current" />
      {downloading === null
        ? "Thinking…"
        : `Downloading ${Math.round(downloading * 100)}%`}
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
  seconds,
  onReload,
  onDelete
}: {
  text: string
  canRegenerate: boolean
  seconds?: number
  onReload: () => void
  onDelete?: () => void
}) {
  return (
    <Actions>
      {text && <CopyAction text={text} />}
      {text && <ReadAloudAction text={text} />}
      {canRegenerate && (
        <Action label="Regenerate response" onPress={onReload}>
          <ArrowRotateRight />
        </Action>
      )}
      {text && (
        <Dropdown>
          <Action label="More actions">
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
      {seconds !== undefined && (
        <span className="px-1 text-xs text-muted tabular-nums">
          {Number(seconds.toFixed(1))}s
        </span>
      )}
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
      label={copied ? "Copied" : "Copy response"}
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
