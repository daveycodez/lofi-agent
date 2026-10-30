/**
 * The `input` widget: the prompt input, with the model picker, the Think
 * toggle and voice input in its toolbar.
 */
import Bulb from "@gravity-ui/icons/Bulb"
import Microphone from "@gravity-ui/icons/Microphone"
import { Skeleton, ToggleButton, Tooltip } from "@heroui/react"
import { useRef, useState } from "react"

import {
  setSelectedModel,
  setThinking,
  useSelectedModel,
  useThinking
} from "#/chat/model"
import {
  afmStatus,
  canThink,
  checkAfm,
  PROVIDERS,
  useModels
} from "#/chat/models"
import type { ModelId } from "#/chat/models"
import { Action } from "#/components/ai/actions"
import { ModelSelector } from "#/components/ai/model-selector"
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools
} from "#/components/ai/prompt-input"
import { useDictation } from "#/lib/dictation"
import { AppleSetupModal } from "./apple-setup-modal"
import { useComposer, useThread } from "./thread-context"
import { useChatContext } from "./ui-components"

export function ChatPromptInput() {
  const { value, setValue } = useComposer()
  const { send } = useThread()
  const { isLoading, stop } = useChatContext()
  const model = useSelectedModel()
  // A message sent while a reply runs waits in the queue.
  const canSend = !!value.trim()

  // Dictation adds to whatever was already typed when it started.
  const dictation = useDictation()
  const toggleDictation = () => {
    if (dictation.isListening) return dictation.stop()
    const before = value.trimEnd()
    dictation.start((text) =>
      setValue(before ? `${before} ${text.trimStart()}` : text.trimStart())
    )
  }

  const submit = () => {
    if (!canSend) return
    dictation.cancel()
    send(value.trim())
    setValue("")
  }

  return (
    <PromptInput label="Message input" name="message">
      <PromptInputTextarea
        placeholder="Send a message..."
        value={value}
        enterKeyHint="send"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <PromptInputToolbar>
        <PromptInputTools>
          <ChatModelSelector />
          {canThink(model) && <ThinkToggle />}
        </PromptInputTools>
        <PromptInputTools end>
          {dictation.isSupported && (
            <Action
              label={
                dictation.isListening ? "Stop voice input" : "Start voice input"
              }
              aria-pressed={dictation.isListening}
              onPress={toggleDictation}
              className={dictation.isListening ? "text-danger" : "text-muted"}
            >
              <Microphone />
            </Action>
          )}
          <PromptInputSubmit
            // Stop while a reply runs, until there's a message to queue.
            status={isLoading && !canSend ? "streaming" : "ready"}
            isDisabled={!canSend}
            onSubmit={submit}
            onStop={stop}
          />
        </PromptInputTools>
      </PromptInputToolbar>
    </PromptInput>
  )
}

/**
 * The prompt input as it looks before the database opens (and on the
 * server): the same box, read-only, with a skeleton for the picked model,
 * which is a saved setting.
 */
export function PromptInputSkeleton() {
  const noop = () => undefined
  return (
    <PromptInput label="Message input" name="message">
      <PromptInputTextarea placeholder="Send a message..." value="" readOnly />
      <PromptInputToolbar>
        <PromptInputTools>
          <div className="flex h-9 items-center px-3 md:h-8">
            <Skeleton className="h-3.5 w-28 rounded-md" />
          </div>
        </PromptInputTools>
        <PromptInputTools end>
          <PromptInputSubmit
            status="ready"
            isDisabled
            onSubmit={noop}
            onStop={noop}
          />
        </PromptInputTools>
      </PromptInputToolbar>
    </PromptInput>
  )
}

/**
 * The model picker. A local model downloads when the first message is sent
 * to it; Apple's model is checked when picked, and setup steps show if it
 * can't be reached.
 */
function ChatModelSelector() {
  const model = useSelectedModel()
  const models = useModels()
  const afm = afmStatus.use()
  // Models by provider, each a section; a provider this device offers no
  // model of is left out.
  const sections = PROVIDERS.map((provider) => ({
    ...provider,
    models: models.filter((m) => m.adapter.name === provider.id)
  })).filter((section) => section.models.length > 0)
  // Counts picks, so a slow Apple check can't undo a newer choice.
  const lastPick = useRef(0)
  // The Apple model while its check runs, so the checkmark moves right away.
  const [pending, setPending] = useState<ModelId | null>(null)
  const [setupOpen, setSetupOpen] = useState(false)

  const select = async (key: ModelId) => {
    const pick = ++lastPick.current
    if (key === "apple-fm") {
      // Only switch once apfel confirms Apple's model is there.
      afmStatus.set("checking")
      setPending(key)
      const isAvailable = await checkAfm()
      // A newer pick happened meanwhile; it owns the status now.
      if (pick !== lastPick.current) return
      setPending(null)
      afmStatus.set(isAvailable ? "idle" : "unavailable")
      setSetupOpen(!isAvailable)
      if (isAvailable) setSelectedModel(key)
      return
    }
    setPending(null)
    afmStatus.set("idle")
    setSelectedModel(key)
  }

  return (
    <>
      <AppleSetupModal
        isOpen={setupOpen}
        onOpenChange={setSetupOpen}
        onRetry={() => void select("apple-fm")}
        isChecking={afm === "checking"}
      />
      <ModelSelector
        sections={sections}
        value={pending ?? model}
        onChange={(id) => void select(id as ModelId)}
        status={
          afm === "checking"
            ? { text: "Loading model" }
            : afm === "unavailable"
              ? { text: "Apple model not available", isError: true }
              : null
        }
      />
    </>
  )
}

/** Turns on reasoning before answering, for models that can think. */
function ThinkToggle() {
  const isThinking = useThinking()
  return (
    <Tooltip delay={300}>
      <ToggleButton
        isIconOnly
        size="sm"
        variant="ghost"
        isSelected={isThinking}
        onChange={setThinking}
        aria-label="Think before answering"
        className={isThinking ? "shrink-0" : "shrink-0 text-muted"}
      >
        <Bulb />
      </ToggleButton>
      <Tooltip.Content>
        {isThinking
          ? "Thinking on: slower, more careful"
          : "Think before answering"}
      </Tooltip.Content>
    </Tooltip>
  )
}
