/**
 * A chat as plain text, for pasting somewhere to see what the model did:
 * every message in order, with the assistant's thinking, its tool calls and
 * their results.
 */
import type { MessagePart, UIMessage } from "@tanstack/ai"

type ToolResult = Extract<MessagePart, { type: "tool-result" }>

const ROLES: Record<UIMessage["role"], string> = {
  user: "User",
  assistant: "Assistant",
  system: "System",
  activity: "Activity"
}

/** JSON on one line, or the text as it is if it isn't JSON. */
function compactJson(json: string) {
  try {
    return JSON.stringify(JSON.parse(json))
  } catch {
    return json.trim()
  }
}

const thinkingBlock = (text: string) => `<thinking>\n${text}\n</thinking>`

function resultText(content: ToolResult["content"]) {
  if (typeof content === "string") return content
  return content
    .map((p) => (p.type === "text" ? p.content : `[${p.type}]`))
    .join("\n")
}

/** A tool call and how it ended: its output, its error, or nothing yet. */
function toolCallBlock(
  part: Extract<MessagePart, { type: "tool-call" }>,
  result: ToolResult | undefined
) {
  const lines = [
    `Tool call: ${part.name} ${compactJson(part.arguments || "{}")}`
  ]
  if (result && (result.error || result.state === "error")) {
    lines.push(`Error: ${result.error ?? resultText(result.content)}`)
  } else if (part.output !== undefined) {
    lines.push(
      "Result:",
      typeof part.output === "string"
        ? part.output
        : JSON.stringify(part.output, null, 2)
    )
  } else if (result) {
    lines.push("Result:", resultText(result.content))
  } else if (part.state === "error") {
    lines.push("Error: The tool failed.")
  } else {
    lines.push("(no result)")
  }
  return lines.join("\n")
}

function partBlocks(
  part: MessagePart,
  results: Map<string, ToolResult>,
  shownResults: Set<string>
): Array<string> {
  switch (part.type) {
    case "text":
      return part.content.trim() ? [part.content] : []
    case "thinking":
      return part.content.trim() ? [thinkingBlock(part.content.trim())] : []
    case "tool-call": {
      const result = results.get(part.id)
      if (result) shownResults.add(part.id)
      return [toolCallBlock(part, result)]
    }
    case "tool-result":
      // Shown with its call; only a result with no call gets its own block.
      return shownResults.has(part.toolCallId)
        ? []
        : [`Tool result (${part.toolCallId}):\n${resultText(part.content)}`]
    default:
      return [`[${part.type}]`]
  }
}

/** The whole chat as Markdown-ish text, one `## Role` section per turn. */
export function formatChatLog(messages: ReadonlyArray<UIMessage>) {
  const results = new Map<string, ToolResult>()
  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type === "tool-result") results.set(part.toolCallId, part)
    }
  }
  const shownResults = new Set<string>()
  const sections: Array<string> = []
  let lastRole: UIMessage["role"] | undefined
  for (const message of messages) {
    const blocks = message.parts.flatMap((part) =>
      partBlocks(part, results, shownResults)
    )
    if (blocks.length === 0) continue
    // A reply that called tools spans several assistant messages: one section.
    if (message.role !== lastRole) sections.push(`## ${ROLES[message.role]}`)
    lastRole = message.role
    sections.push(...blocks)
  }
  return sections.join("\n\n")
}

/**
 * Copies text that may still be loading. Safari only allows a clipboard
 * write during the tap, and an `await` (reading IndexedDB, say) ends that,
 * but it accepts a ClipboardItem holding a promise of the text. Call this
 * straight from the tap handler, before awaiting anything.
 */
export async function copyText(text: string | Promise<string>) {
  if (typeof ClipboardItem !== "undefined") {
    const blob = Promise.resolve(text).then(
      (t) => new Blob([t], { type: "text/plain" })
    )
    try {
      await navigator.clipboard.write([
        new ClipboardItem({ "text/plain": blob })
      ])
      return
    } catch {
      // Some browsers don't take a promise in a ClipboardItem: plain text,
      // which they allow for a few seconds after the tap.
    }
  }
  await navigator.clipboard.writeText(await text)
}
