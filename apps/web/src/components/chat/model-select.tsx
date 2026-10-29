import ChevronDown from "@gravity-ui/icons/ChevronDown"
import {
  Button,
  ListBox,
  Popover,
  ProgressCircle,
  Spinner
} from "@heroui/react"
import { useRef, useState } from "react"

import {
  afmStatus,
  checkAfm,
  loadModel,
  modelInfo,
  modelStore,
  useCurrentModel,
  useModels,
  useModelStatus
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
  const local = useModelStatus(model)
  const [isOpen, setIsOpen] = useState(false)
  const current = modelInfo(model)
  // Counts picks, so a slow Apple check can't undo a newer choice.
  const lastPick = useRef(0)
  // The Apple model while its check runs, so the checkmark moves right away.
  const [pending, setPending] = useState<ModelId | null>(null)

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
      if (isAvailable) modelStore.set(key)
      return
    }
    setPending(null)
    afmStatus.set("idle")
    modelStore.set(key)
    // Start loading (downloading the first time) as soon as it's picked.
    loadModel(key)
  }

  // `progress` is 0-1 for a determinate download, null to just show activity.
  const status: {
    text: string
    progress?: number | null
    isError?: boolean
  } | null =
    afm === "checking"
      ? { text: "Loading model", progress: null }
      : afm === "unavailable"
        ? { text: "Apple model not available", isError: true }
        : local.state === "loading"
          ? !local.cached && local.progress < 1
            ? {
                text: `Downloading ${Math.round(local.progress * 100)}%`,
                progress: local.progress
              }
            : { text: "Loading model", progress: null }
          : local.state === "error"
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
          className={`flex items-center gap-1.5 text-xs whitespace-nowrap tabular-nums ${status.isError ? "text-danger" : "text-muted"}`}
        >
          {status.progress === null && <Spinner size="sm" color="current" />}
          {typeof status.progress === "number" && (
            <ProgressCircle
              aria-label={status.text}
              size="sm"
              value={status.progress * 100}
            >
              <ProgressCircle.Track>
                <ProgressCircle.TrackCircle />
                <ProgressCircle.FillCircle className="stroke-current" />
              </ProgressCircle.Track>
            </ProgressCircle>
          )}
          {status.text}
        </span>
      )}
    </div>
  )
}
