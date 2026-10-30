import type { ToolCallPart, ToolResultPart } from "@tanstack/ai"
import type { InterruptProps, ToolProps } from "@tanstack/ai-react/ui"
import { useContext } from "react"
import type { ComponentType } from "react"

import type { chatOptions } from "#/chat/options"
import {
  Tool,
  ToolApproval,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput
} from "#/components/ai/tool"
import { ReplyContext } from "./thread-context"
import { useChatContext } from "./ui-components"

const asText = (value: unknown) =>
  typeof value === "string" ? value : JSON.stringify(value, null, 2)

/** How a tool call ended: its output, its error, or neither yet. */
function outcomeOf(part: ToolCallPart, result?: ToolResultPart) {
  const error =
    result && (result.error || result.state === "error")
      ? (result.error ?? asText(result.content))
      : part.state === "error"
        ? "The tool failed."
        : undefined
  const output =
    part.output !== undefined
      ? asText(part.output)
      : result && typeof result.content === "string"
        ? result.content
        : undefined
  return { error, output }
}

/** Any MCP server's tool: its name, with its arguments and result inside. */
export function McpTool({ part, result }: ToolProps<typeof chatOptions>) {
  const { isRunning } = useContext(ReplyContext)
  const { error, output } = outcomeOf(part, result)
  const status = error
    ? "error"
    : part.state === "approval-requested"
      ? "approval"
      : output === undefined && isRunning
        ? "running"
        : "done"
  return (
    <Tool>
      <ToolHeader name={part.name} status={status} />
      <ToolContent>
        <ToolInput input={part.arguments} />
        <ToolOutput output={output} errorText={error} />
      </ToolContent>
    </Tool>
  )
}

/**
 * Asks before an MCP tool runs (src/lib/mcp.ts decides which do). It shows
 * in the layout's `Interrupts` list, as in TanStack's example; the tool's
 * card above shows the call waiting.
 */
export function McpToolApproval({
  interrupt
}: InterruptProps<typeof chatOptions>) {
  const { resuming } = useChatContext()
  if (interrupt.kind !== "tool-approval") return null
  return (
    <ToolApproval
      name={interrupt.toolName}
      input={JSON.stringify(interrupt.originalArgs ?? {})}
      status={interrupt.status}
      isDisabled={!interrupt.canResolve || resuming}
      onApprove={() => interrupt.resolveInterrupt(true)}
      onDeny={() => interrupt.resolveInterrupt(false)}
    />
  )
}

/** Keys that tooling probes on any object; they must not look like tools. */
const NOT_TOOLS = new Set(["then", "toJSON", "constructor", "$$typeof"])

/**
 * A `toolsComponents` map that gives every tool name the same widget. The
 * chat kit renders a tool call only through a widget registered under its
 * exact name, with no fallback (TanStack/ai#1953), and MCP tools are only
 * known at runtime. The kit reads this map by name when it renders and
 * lists no keys of it, so tool approvals are not placed inline (this app
 * has none).
 */
export function everyTool<TProps>(
  widget: ComponentType<TProps>
): Record<string, ComponentType<TProps>> {
  return new Proxy(
    {},
    {
      get: (_target, key) =>
        typeof key === "string" && !NOT_TOOLS.has(key) ? widget : undefined
    }
  )
}
