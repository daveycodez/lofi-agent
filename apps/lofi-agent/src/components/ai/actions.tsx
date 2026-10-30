import { Button, Tooltip } from "@heroui/react"
import type { ComponentProps, ReactNode } from "react"

/**
 * A row of actions under a message, faded to half until the message is
 * hovered, one of them has focus or its menu is open, as in HeroUI Pro's
 * ChatMessage.
 */
export function Actions({
  className = "",
  ...props
}: ComponentProps<"div">) {
  return (
    <div
      className={`flex items-center gap-1 opacity-50 transition-opacity group-hover/message:opacity-100 focus-within:opacity-100 has-aria-expanded:opacity-100 ${className}`}
      {...props}
    />
  )
}

export type ActionProps = Omit<
  ComponentProps<typeof Button>,
  "children" | "isIconOnly"
> & {
  /** The tooltip and accessible name. */
  label: string
  children: ReactNode
}

/** An icon-only ghost button with a tooltip. */
export function Action({ label, children, ...props }: ActionProps) {
  return (
    <Tooltip delay={300}>
      <Button
        isIconOnly
        size="sm"
        variant="ghost"
        aria-label={label}
        className="text-muted"
        {...props}
      >
        {children}
      </Button>
      <Tooltip.Content>{label}</Tooltip.Content>
    </Tooltip>
  )
}
