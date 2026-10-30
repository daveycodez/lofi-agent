import type { PartProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

/**
 * Parts this app doesn't show: a tool result shows with its call, and the
 * local models send no images, audio or documents.
 */
export function FallbackPart(_props: PartProps<typeof chatOptions>) {
  return null
}
