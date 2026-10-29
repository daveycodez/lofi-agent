import ChevronDown from "@gravity-ui/icons/ChevronDown"
import { Button, ListBox, Popover, Spinner } from "@heroui/react"
import { useRef, useState } from "react"

import {
  afmStatus,
  checkAfm,
  modelInfo,
  modelStore,
  useCurrentModel,
  useModels
} from "#/lib/llm/models"
import type { ModelId } from "#/lib/llm/models"
import { AppleSetupModal } from "#/components/chat/apple-setup-modal"

/** Provider logo. The SVGs are drawn in black, so invert them in dark mode. */
function ModelLogo({ src }: { src: string }) {
  return (
    <img
      src={src}
      alt=""
      aria-hidden
      className="size-3.5 shrink-0 dark:invert"
    />
  )
}

/**
 * Model picker for the composer's button row: a ghost Button opening a
 * Popover with a single-selection ListBox (checkmark on the right).
 */
export function ModelSelect() {
  const model = useCurrentModel()
  const models = useModels()
  const afm = afmStatus.use()
  const [isOpen, setIsOpen] = useState(false)
  const current = modelInfo(model)
  // Counts picks, so a slow Apple check can't undo a newer choice.
  const lastPick = useRef(0)
  // The Apple model while its check runs, so the checkmark moves right away.
  const [pending, setPending] = useState<ModelId | null>(null)
  // Setup steps, shown when Apple's model can't be reached.
  const [setupOpen, setSetupOpen] = useState(false)

  const select = async (key: ModelId) => {
    setIsOpen(false)
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
      if (isAvailable) modelStore.set(key)
      return
    }
    setPending(null)
    afmStatus.set("idle")
    // A local model downloads when the first message is sent to it.
    modelStore.set(key)
  }

  // Apple's model is checked when picked; local models report their
  // download in the reply.
  const status: { text: string; isError?: boolean } | null =
    afm === "checking"
      ? { text: "Loading model" }
      : afm === "unavailable"
        ? { text: "Apple model not available", isError: true }
        : null

  return (
    // min-w-0: a long model name truncates instead of pushing the composer's
    // other buttons out.
    <div className="flex min-w-0 items-center gap-1">
      <AppleSetupModal
        isOpen={setupOpen}
        onOpenChange={setSetupOpen}
        onRetry={() => void select("apple-fm")}
        isChecking={afm === "checking"}
      />
      <Popover isOpen={isOpen} onOpenChange={setIsOpen}>
        <Button
          size="sm"
          variant="ghost"
          aria-label={`Model: ${current.name}`}
          className="min-w-0 text-muted"
        >
          <ModelLogo src={current.icon} />
          <span className="min-w-0 truncate">{current.name}</span>
          {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
          <ChevronDown className="ms-0 shrink-0" />
        </Button>
        <Popover.Content placement="bottom start">
          <Popover.Dialog aria-label="Model" className="p-1">
            <ListBox
              aria-label="Model"
              selectionMode="single"
              disallowEmptySelection
              selectedKeys={new Set([pending ?? model])}
              onSelectionChange={(keys) => {
                const [key] = keys === "all" ? [] : [...keys]
                if (key) void select(key as ModelId)
              }}
            >
              {models.map((m) => (
                <ListBox.Item key={m.id} id={m.id} textValue={m.name}>
                  <span className="flex items-center gap-2">
                    <ModelLogo src={m.icon} />
                    {m.name}
                  </span>
                  <ListBox.ItemIndicator />
                </ListBox.Item>
              ))}
            </ListBox>
          </Popover.Dialog>
        </Popover.Content>
      </Popover>
      {status && (
        <span
          className={`flex min-w-0 items-center gap-1.5 text-xs whitespace-nowrap tabular-nums ${status.isError ? "text-danger" : "text-muted"}`}
        >
          {!status.isError && (
            <Spinner size="sm" color="current" className="shrink-0" />
          )}
          <span className="min-w-0 truncate">{status.text}</span>
        </span>
      )}
    </div>
  )
}
