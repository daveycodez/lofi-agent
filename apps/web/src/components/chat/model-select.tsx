import ChevronDown from "@gravity-ui/icons/ChevronDown"
import { Button, ListBox, Popover } from "@heroui/react"
import { useState } from "react"

import { lfm3Status, loadLfm3 } from "#/lib/llm/lfm3-connection"
import {
  afmStatus,
  checkAfm,
  modelInfo,
  modelStore,
  useCurrentModel,
  useModels
} from "#/lib/llm/models"
import type { ModelId } from "#/lib/llm/models"

/** Provider logo. The SVGs are drawn in black, so invert them in dark mode. */
function ModelLogo({ src }: { src: string }) {
  return <img src={src} alt="" aria-hidden className="size-3.5 dark:invert" />
}

/**
 * Model picker for the composer's button row: a ghost Button opening a
 * Popover with a single-selection ListBox (checkmark on the right).
 */
export function ModelSelect() {
  const model = useCurrentModel()
  const models = useModels()
  const afm = afmStatus.use()
  const lfm3 = lfm3Status.use()
  const [isOpen, setIsOpen] = useState(false)
  const current = modelInfo(model)

  const select = async (key: ModelId) => {
    setIsOpen(false)
    if (key === "apple-fm") {
      // Only switch once apfel confirms Apple's model is there.
      if (await checkAfm()) modelStore.set(key)
      return
    }
    afmStatus.set("idle")
    modelStore.set(key)
    // Start the one-time download as soon as LFM3 is picked.
    void loadLfm3().catch(() => undefined)
  }

  const status: { text: string; isError?: boolean } | null =
    afm === "checking"
      ? { text: "Connecting to Apple model…" }
      : afm === "unavailable"
        ? { text: "Apple model not available", isError: true }
        : model === "lfm3" && lfm3.state === "loading"
          ? { text: `Downloading ${Math.round(lfm3.progress * 100)}%` }
          : model === "lfm3" && lfm3.state === "error"
            ? { text: "Couldn't load", isError: true }
            : null

  return (
    <div className="flex items-center gap-2">
      <Popover isOpen={isOpen} onOpenChange={setIsOpen}>
        <Button
          size="sm"
          variant="ghost"
          aria-label={`Model: ${current.name}`}
          className="text-muted"
        >
          <ModelLogo src={current.icon} />
          {current.name}
          <ChevronDown />
        </Button>
        <Popover.Content placement="bottom start">
          <Popover.Dialog aria-label="Model" className="p-1">
            <ListBox
              aria-label="Model"
              selectionMode="single"
              disallowEmptySelection
              selectedKeys={new Set([model])}
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
          className={`text-xs tabular-nums ${status.isError ? "text-danger" : "text-muted"}`}
        >
          {status.text}
        </span>
      )}
    </div>
  )
}
