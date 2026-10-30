import type { InterruptProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

/**
 * An interrupt with no widget of its own: one this chat can't answer
 * (paused elsewhere), or a generic pause this app doesn't define.
 */
export function FallbackInterrupt({
  interrupt
}: InterruptProps<typeof chatOptions>) {
  return (
    <p className="text-sm text-muted">
      {interrupt.kind === "unbound"
        ? `Paused elsewhere: ${interrupt.reason}`
        : interrupt.reason}
    </p>
  )
}
