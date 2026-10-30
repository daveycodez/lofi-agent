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
import type { ToolStatus } from "#/components/ai/tool"
import { ReplyContext } from "./thread-context"
import { useChatContext } from "./ui-components"

const asText = (value: unknown) =>
  typeof value === "string" ? value : JSON.stringify(value, null, 2)

/** Why a call failed, if it did. */
function errorOf(part: ToolCallPart, result?: ToolResultPart) {
  if (result && (result.error || result.state === "error")) {
    return result.error ?? asText(result.content)
  }
  return part.state === "error" ? "The tool failed." : undefined
}

/** Where a call is, from its state, as TanStack's example reads it. */
function statusOf(
  part: ToolCallPart,
  result: ToolResultPart | undefined,
  isRunning: boolean
): ToolStatus {
  if (errorOf(part, result)) return "error"
  if (part.state === "approval-requested") return "approval"
  if (part.output !== undefined || result) return "done"
  if (part.state === "awaiting-input" || part.state === "input-streaming") {
    return isRunning ? "preparing" : "stopped"
  }
  return isRunning ? "running" : "stopped"
}

/** Any MCP server's tool: its name and status, its arguments and result inside. */
export function McpTool({ part, result }: ToolProps<typeof chatOptions>) {
  const { isRunning } = useContext(ReplyContext)
  const errorText = errorOf(part, result)
  const output =
    part.output ?? (typeof result?.content === "string" ? result.content : undefined)
  return (
    <Tool>
      <ToolHeader name={part.name} status={statusOf(part, result, isRunning)} />
      <ToolContent>
        <ToolInput input={part.input ?? part.arguments} />
        <ToolOutput output={output} errorText={errorText} />
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
      input={interrupt.originalArgs}
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
