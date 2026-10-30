/**
 * HeroUI's prompt input: a TextField and InputGroup holding a growing
 * TextArea and a toolbar of buttons.
 */
import ArrowUp from "@gravity-ui/icons/ArrowUp"
import StopFill from "@gravity-ui/icons/StopFill"
import { Button, InputGroup, Kbd, TextField, Tooltip } from "@heroui/react"
import { useLayoutEffect, useRef } from "react"
import type { ComponentProps, ReactNode } from "react"

export function PromptInput({
  label,
  name,
  className,
  children
}: {
  /** The accessible name. */
  label: string
  name?: string
  className?: string
  children: ReactNode
}) {
  return (
    <TextField fullWidth aria-label={label} name={name} className={className}>
      <InputGroup
        fullWidth
        className="flex flex-col gap-3 rounded-3xl pt-3.5 pb-2"
      >
        {children}
      </InputGroup>
    </TextField>
  )
}

/** Grows with its text up to `maxHeight` pixels, then scrolls. */
export function PromptInputTextarea({
  value,
  maxHeight = 192,
  ...props
}: Omit<ComponentProps<typeof InputGroup.TextArea>, "value" | "ref"> & {
  value: string
  maxHeight?: number
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden"
  }, [value, maxHeight])
  return (
    <InputGroup.TextArea
      ref={ref}
      rows={1}
      value={value}
      className="min-h-0 w-full flex-none resize-none px-3.5 py-0"
      {...props}
    />
  )
}

export function PromptInputToolbar({ children }: { children: ReactNode }) {
  return (
    <InputGroup.Suffix className="flex w-full items-center gap-1 px-3 py-0">
      {children}
    </InputGroup.Suffix>
  )
}

/** A group of toolbar buttons. `end` pushes it to the right. */
export function PromptInputTools({
  end = false,
  children
}: {
  end?: boolean
  children: ReactNode
}) {
  return (
    <div
      className={`flex items-center gap-1 ${end ? "ms-auto shrink-0" : "min-w-0"}`}
    >
      {children}
    </div>
  )
}

/** Sends the message, or stops the reply while one is running. */
export function PromptInputSubmit({
  status,
  isDisabled,
  onSubmit,
  onStop
}: {
  status: "ready" | "streaming"
  isDisabled: boolean
  onSubmit: () => void
  onStop: () => void
}) {
  if (status === "streaming") {
    return (
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
    )
  }
  return (
    <Tooltip delay={300}>
      <Button
        isIconOnly
        size="sm"
        aria-label="Send message"
        isDisabled={isDisabled}
        onPress={onSubmit}
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
  )
}
