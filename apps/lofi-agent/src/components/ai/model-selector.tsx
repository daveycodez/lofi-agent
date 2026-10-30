import ChevronDown from "@gravity-ui/icons/ChevronDown"
import {
  Button,
  Header,
  ListBox,
  Popover,
  Spinner,
  ToggleButton,
  ToggleButtonGroup
} from "@heroui/react"
import { useState } from "react"

export type ModelOption = { id: string; name: string; icon: string }

/** A thinking effort level the picked model offers. */
export type EffortOption = { id: string; name: string }

export type ModelSection = {
  id: string
  name: string
  models: Array<ModelOption>
}

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
 * A model picker for the prompt input's toolbar: a tertiary Button opening a
 * Popover with a single-selection ListBox, one section per provider. For a
 * model with thinking effort levels, a "Thinking" row under the list picks
 * one, and the button shows it after the model's name (as assistant-ui's
 * model selector does).
 */
export function ModelSelector({
  sections,
  value,
  onChange,
  efforts = [],
  effort,
  effortLabel,
  onEffortChange,
  status
}: {
  sections: Array<ModelSection>
  value: string
  onChange: (id: string) => void
  /** The picked model's effort levels, least to most; none hides the row. */
  efforts?: Array<EffortOption>
  /** The level the picked model runs at. */
  effort?: string
  /** Shown muted after the model's name, such as "Thinking" or "High". */
  effortLabel?: string
  onEffortChange?: (id: string) => void
  /** A short note beside the button, such as a check in progress. */
  status?: { text: string; isError?: boolean } | null
}) {
  const [isOpen, setIsOpen] = useState(false)
  const current = sections
    .flatMap((section) => section.models)
    .find((model) => model.id === value)

  return (
    // min-w-0: a long model name truncates instead of pushing the toolbar's
    // other buttons out.
    <div className="flex min-w-0 items-center gap-1">
      <Popover isOpen={isOpen} onOpenChange={setIsOpen}>
        <Button
          size="sm"
          variant="tertiary"
          aria-label={`Model: ${current?.name ?? value}${effortLabel ? ` (${effortLabel})` : ""}`}
          // The toolbar it sits in is muted: the model's name isn't, only
          // its effort.
          className="min-w-0 text-foreground"
        >
          {current && <ModelLogo src={current.icon} />}
          <span className="min-w-0 truncate">{current?.name ?? value}</span>
          {effortLabel && (
            <span className="shrink-0 text-muted">{effortLabel}</span>
          )}
          {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
          <ChevronDown className="ms-0 shrink-0" />
        </Button>
        <Popover.Content placement="bottom start">
          {/* Within the room the popover gets: the list scrolls, the
              Thinking row stays in view under it. */}
          <Popover.Dialog
            aria-label="Model"
            className="flex max-h-[inherit] flex-col p-1"
          >
            <ListBox
              aria-label="Model"
              // Scrolls right up to the Thinking row's line; its own padding
              // is the gap under the last model.
              className="min-h-0 overflow-y-auto"
              selectionMode="single"
              disallowEmptySelection
              selectedKeys={new Set([value])}
              onSelectionChange={(keys) => {
                const [key] = keys === "all" ? [] : [...keys]
                if (key === undefined) return
                setIsOpen(false)
                onChange(String(key))
              }}
            >
              {sections.map((section) => (
                <ListBox.Section key={section.id}>
                  <Header>{section.name}</Header>
                  {section.models.map((model) => (
                    <ListBox.Item
                      key={model.id}
                      id={model.id}
                      textValue={model.name}
                    >
                      <span className="flex items-center gap-2">
                        <ModelLogo src={model.icon} />
                        {model.name}
                      </span>
                      <ListBox.ItemIndicator />
                    </ListBox.Item>
                  ))}
                </ListBox.Section>
              ))}
            </ListBox>
            {efforts.length > 0 && (
              <div className="flex shrink-0 items-center justify-between gap-3 border-t border-separator px-2 pt-2 pb-1">
                <span className="text-xs text-muted">Thinking</span>
                <ToggleButtonGroup
                  aria-label="Thinking effort"
                  size="sm"
                  isDetached
                  selectionMode="single"
                  disallowEmptySelection
                  selectedKeys={effort ? [effort] : []}
                  onSelectionChange={(keys) => {
                    const [key] = [...keys]
                    if (key !== undefined) onEffortChange?.(String(key))
                  }}
                  className="gap-0.5"
                >
                  {efforts.map((option) => (
                    <ToggleButton
                      key={option.id}
                      id={option.id}
                      variant="ghost"
                      className="h-6 min-w-0 rounded-md px-2 text-xs font-normal text-muted data-[selected=true]:bg-default data-[selected=true]:font-medium data-[selected=true]:text-foreground"
                    >
                      {option.name}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
              </div>
            )}
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
