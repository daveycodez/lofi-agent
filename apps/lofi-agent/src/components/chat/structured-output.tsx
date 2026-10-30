import type { PartProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

/** JSON indented two spaces; a string that isn't JSON (yet) as it is. */
function indented(data: unknown) {
  if (typeof data !== "string") return JSON.stringify(data, null, 2)
  try {
    return JSON.stringify(JSON.parse(data), null, 2)
  } catch {
    return data
  }
}

/**
 * A reply written as structured data rather than text. This app asks for
 * none, so it shows the data as it comes, as JSON.
 */
export function StructuredOutputPart({
  part
}: PartProps<typeof chatOptions, "structuredOutput">) {
  if (part.errorMessage) {
    return <p className="text-sm text-danger">{part.errorMessage}</p>
  }
  const data = part.partial ?? part.raw
  return (
    // The same bordered box as a tool call.
    <div className="flex flex-col gap-2 rounded-xl border border-border p-3">
      <p className="text-xs text-muted">Structured output</p>
      <pre className="max-h-80 overflow-auto font-mono text-[11px] leading-5 whitespace-pre-wrap">
        {indented(data)}
      </pre>
    </div>
  )
}
