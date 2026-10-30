import ChevronDown from "@gravity-ui/icons/ChevronDown"
import { Button, Header, ListBox, Popover, Spinner } from "@heroui/react"
import { useState } from "react"

export type ModelOption = { id: string; name: string; icon: string }

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
 * A model picker for the prompt input's toolbar: a ghost Button opening a
 * Popover with a single-selection ListBox, one section per provider.
 */
export function ModelSelector({
  sections,
  value,
  onChange,
  status
}: {
  sections: Array<ModelSection>
  value: string
  onChange: (id: string) => void
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
          variant="ghost"
          aria-label={`Model: ${current?.name ?? value}`}
          className="min-w-0 text-muted"
        >
          {current && <ModelLogo src={current.icon} />}
          <span className="min-w-0 truncate">{current?.name ?? value}</span>
          {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
          <ChevronDown className="ms-0 shrink-0" />
        </Button>
        <Popover.Content placement="bottom start">
          <Popover.Dialog aria-label="Model" className="p-1">
            <ListBox
              aria-label="Model"
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
