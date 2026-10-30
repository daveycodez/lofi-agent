/**
 * Tool calls Apple's model writes as text. apfel turns the model's tool
 * calls into OpenAI `tool_calls`, but when it doesn't recognize one it
 * passes it on as text instead, like
 * `{"tool_calls": [{"function": "get_weather", "arguments": {…}}]}`.
 */

export type ParsedToolCall = { name: string; args: Record<string, unknown> }

/** Where Apple's model starts a tool call it writes as text. */
export const JSON_TOOL_CALLS_MARKER = '{"tool_calls"'

/** The `{…}` that starts at `text[start]`, braces in strings ignored. */
function balancedObject(text: string, start: number) {
  let depth = 0
  let inString = false
  for (let i = start; i < text.length; i++) {
    const c = text[i]
    if (inString) {
      if (c === "\\") i++
      else if (c === '"') inString = false
    } else if (c === '"') inString = true
    else if (c === "{") depth++
    else if (c === "}" && --depth === 0) return text.slice(start, i + 1)
  }
  return null
}

/** JSON arguments, read leniently: stray strings without a key are dropped. */
function parseJsonArgs(raw: string): Record<string, unknown> | null {
  for (const candidate of [raw, raw.replace(/,\s*"[^"]*"\s*(?=[,}])/g, "")]) {
    try {
      const value = JSON.parse(candidate) as unknown
      const args = typeof value === "string" ? JSON.parse(value) : value
      if (args && typeof args === "object") {
        return args as Record<string, unknown>
      }
    } catch {
      // Try the next reading.
    }
  }
  return null
}

/**
 * Tool calls Apple's model wrote as JSON text. Its JSON varies and is often
 * broken (`"function": "name"`, `"name: "name"`, stray strings in the
 * arguments), so instead of parsing the whole thing, this finds each quoted
 * name of a tool that exists and reads the `"arguments"` object after it.
 */
export function parseJsonToolCalls(
  text: string,
  toolNames: Iterable<string>
): Array<ParsedToolCall> {
  const mentions: Array<{ name: string; at: number }> = []
  for (const name of toolNames) {
    const quoted = `"${name}"`
    for (
      let at = text.indexOf(quoted);
      at >= 0;
      at = text.indexOf(quoted, at + 1)
    ) {
      mentions.push({ name, at })
    }
  }
  mentions.sort((a, b) => a.at - b.at)
  const calls: Array<ParsedToolCall> = []
  mentions.forEach(({ name, at }, i) => {
    // The arguments belong to the last mention of a name before them. The
    // model sometimes writes a broken "arguments" and then a good one: the
    // first that reads is used.
    const part = text.slice(at, mentions[i + 1]?.at ?? text.length)
    for (const key of part.matchAll(/"arguments"?\s*:\s*/g)) {
      const from = key.index + key[0].length
      const raw =
        part[from] === "{"
          ? balancedObject(part, from)
          : /^"(?:[^"\\]|\\.)*"/.exec(part.slice(from))?.[0]
      const args = raw ? parseJsonArgs(raw) : null
      if (args) {
        calls.push({ name, args })
        return
      }
    }
  })
  return calls
}

/** Length of the longest end of `text` that could start `marker`. */
function partialMarker(text: string, marker: string) {
  for (let n = Math.min(text.length, marker.length); n > 0; n--) {
    if (marker.startsWith(text.slice(-n))) return n
  }
  return 0
}

/**
 * Splits a streaming answer into the text before the first tool call written
 * as text (passed through as it arrives) and the tool-call JSON after it
 * (held back and parsed once the reply ends).
 */
export function createToolCallSplitter(marker = JSON_TOOL_CALLS_MARKER) {
  let held = ""
  let markup: string | null = null

  /** Text that can be shown now. */
  function push(text: string) {
    if (markup !== null) {
      markup += text
      return ""
    }
    held += text
    const at = held.indexOf(marker)
    if (at >= 0) {
      const before = held.slice(0, at)
      markup = held.slice(at)
      held = ""
      return before
    }
    const keep = partialMarker(held, marker)
    const out = held.slice(0, held.length - keep)
    held = held.slice(out.length)
    return out
  }

  /** The rest of the text, and the tool-call JSON if any. */
  function flush() {
    const rest = held
    held = ""
    return { text: rest, markup }
  }

  return { push, flush }
}
