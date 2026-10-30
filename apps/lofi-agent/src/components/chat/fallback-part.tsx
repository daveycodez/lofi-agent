import type { PartProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

/** A part with no widget of its own. */
export function FallbackPart({ part }: PartProps<typeof chatOptions>) {
  return <p className="text-xs text-muted">Unmapped part: {part.type}</p>
}
