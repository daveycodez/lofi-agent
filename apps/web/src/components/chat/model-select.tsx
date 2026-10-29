import ChevronDown from "@gravity-ui/icons/ChevronDown"
import { Button, ListBox, Popover, Skeleton } from "@heroui/react"
import { useEffect, useState } from "react"

import { lfm3Status, loadLfm3 } from "#/lib/llm/lfm3-connection"
import {
  checkAfm,
  modelInfo,
  modelStore,
  useAvailableModels,
  useCurrentModel
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
  const models = useAvailableModels()
  const lfm3 = lfm3Status.use()
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    void checkAfm()
  }, [])

  // Still checking whether Apple's model is there.
  if (!model) return <Skeleton className="h-8 w-40 rounded-full" />

  const current = modelInfo(model)

  const status =
    model === "lfm3" && lfm3.state === "loading"
      ? `Downloading ${Math.round(lfm3.progress * 100)}%`
      : model === "lfm3" && lfm3.state === "error"
        ? "Couldn't load"
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
                if (!key) return
                modelStore.set(key as ModelId)
                setIsOpen(false)
                // Start the one-time download as soon as LFM3 is picked.
                if (key === "lfm3") void loadLfm3().catch(() => undefined)
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
          className={`text-xs tabular-nums ${lfm3.state === "error" ? "text-danger" : "text-muted"}`}
        >
          {status}
        </span>
      )}
    </div>
  )
}
