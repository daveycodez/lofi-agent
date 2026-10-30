import { Card } from "@heroui/react"
import type { PartProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

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
    <Card className="gap-2 p-3">
      <Card.Header className="p-0">
        <Card.Description className="text-xs">Structured output</Card.Description>
      </Card.Header>
      <Card.Content className="p-0">
        <pre className="max-h-80 overflow-auto font-mono text-xs whitespace-pre-wrap">
          {typeof data === "string" ? data : JSON.stringify(data, null, 2)}
        </pre>
      </Card.Content>
    </Card>
  )
}
