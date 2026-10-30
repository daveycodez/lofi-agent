/**
 * Reading and writing tool calls the way LFM2 and LFM2.5 write them:
 * Pythonic calls between markers,
 * `<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>`.
 *
 * Copied from @tanstack-ai-browser/transformers-js (tool-calls.ts, its LFM2
 * parts), so this package doesn't depend on that one and its worker.
 */

export type ParsedToolCall = { name: string; args: Record<string, unknown> }

/** How a model writes tool calls. */
export type ToolFormat = "lfm2"

export const TOOL_CALL_START = "<|tool_call_start|>"
export const TOOL_CALL_END = "<|tool_call_end|>"

/** A Python-style or JSON value: "x", 'x', 1.5, True, None, [1, 2]… */
function parseValue(raw: string): unknown {
  const s = raw.trim()
  if (/^(["']).*\1$/s.test(s)) {
    try {
      return JSON.parse(s[0] === "'" ? `"${s.slice(1, -1)}"` : s)
    } catch {
      return s.slice(1, -1)
    }
  }
  if (s === "True" || s === "true") return true
  if (s === "False" || s === "false") return false
  if (s === "None" || s === "null") return null
  if (s !== "" && !Number.isNaN(Number(s))) return Number(s)
  try {
    return JSON.parse(s)
  } catch {
    return s
  }
}

/** After a value: the next argument (`, name=`), or the end of the call. */
const VALUE_END = /^\s*(?:,\s*[A-Za-z_]\w*\s*=|,?\s*\)|$)/

/**
 * Where the string starting at `body[start]` ends (the index of its closing
 * quote). The model doesn't escape quotes inside its strings ('JWST's
 * findings'), so a quote only closes the string when the argument ends
 * after it; otherwise it's part of the text.
 */
function stringEnd(body: string, start: number) {
  const quote = body[start]
  let last = -1
  for (let i = start + 1; i < body.length; i++) {
    if (body[i] === "\\") i++
    else if (body[i] === quote) {
      if (VALUE_END.test(body.slice(i + 1))) return i
      last = i
    }
  }
  return last
}

/** A Python string literal's text, `\'` and `\"` included. */
function stringText(raw: string) {
  const inner = raw.slice(1, -1)
  try {
    const escaped = inner.replace(/\\'/g, "'").replace(/(^|[^\\])"/g, '$1\\"')
    return JSON.parse(`"${escaped}"`) as string
  } catch {
    return inner
  }
}

/** Where a `[…]` or `{…}` starting at `body[start]` ends, strings skipped. */
function bracketEnd(body: string, start: number) {
  let depth = 0
  for (let i = start; i < body.length; i++) {
    const c = body[i]
    if (c === '"' || c === "'") {
      const end = stringEnd(body, i)
      if (end < 0) return -1
      i = end
    } else if (c === "[" || c === "{") depth++
    else if ((c === "]" || c === "}") && --depth === 0) return i
  }
  return -1
}

/** `[name(a="x", b=2), other()]` → calls. */
function parsePythonic(body: string) {
  const calls: Array<ParsedToolCall> = []
  const callStart = /([A-Za-z_][\w.-]*)\(/g
  let m: RegExpExecArray | null
  while ((m = callStart.exec(body))) {
    const args: Record<string, unknown> = {}
    let i = m.index + m[0].length
    for (;;) {
      const key = /^\s*([A-Za-z_]\w*)\s*=\s*/.exec(body.slice(i))
      if (!key) break
      i += key[0].length
      const first = body[i]
      let end: number
      if (first === '"' || first === "'") {
        end = stringEnd(body, i)
        if (end < 0) break
        args[key[1]] = stringText(body.slice(i, end + 1))
      } else if (first === "[" || first === "{") {
        end = bracketEnd(body, i)
        if (end < 0) break
        args[key[1]] = parseValue(body.slice(i, end + 1))
      } else {
        const rest = /^[^,)]*/.exec(body.slice(i))?.[0] ?? ""
        end = i + rest.length - 1
        args[key[1]] = parseValue(rest)
      }
      i = end + 1
      const comma = /^\s*,/.exec(body.slice(i))
      if (!comma) break
      i += comma[0].length
    }
    calls.push({ name: m[1], args })
    // The next call starts after this one's closing parenthesis.
    const close = body.indexOf(")", i)
    callStart.lastIndex = close < 0 ? body.length : close + 1
  }
  return calls
}

/** Every tool call in `text`. */
export function parseToolCalls(text: string): Array<ParsedToolCall> {
  const calls: Array<ParsedToolCall> = []
  for (const m of text.matchAll(
    /<\|tool_call_start\|>([\s\S]*?)(?:<\|tool_call_end\|>|$)/g
  )) {
    calls.push(...parsePythonic(m[1]))
  }
  return calls
}

/** Length of the longest end of `text` that could start one of `markers`. */
function partialMarker(text: string, markers: Array<string>) {
  for (let n = Math.min(text.length, 18); n > 0; n--) {
    const end = text.slice(-n)
    if (markers.some((m) => m.startsWith(end))) return n
  }
  return 0
}

/**
 * Splits a streaming answer into the text before the first tool call (passed
 * through as it arrives) and the tool-call markup after it (held back and
 * parsed once the reply ends).
 */
export function createToolCallSplitter(
  markers: Array<string> = [TOOL_CALL_START]
) {
  let held = ""
  let markup: string | null = null

  /** Text that can be shown now. */
  function push(text: string) {
    if (markup !== null) {
      markup += text
      return ""
    }
    held += text
    const at = Math.min(
      ...markers.map((m) => held.indexOf(m)).filter((i) => i >= 0)
    )
    if (Number.isFinite(at)) {
      const before = held.slice(0, at)
      markup = held.slice(at)
      held = ""
      return before
    }
    const keep = partialMarker(held, markers)
    const out = held.slice(0, held.length - keep)
    held = held.slice(out.length)
    return out
  }

  /** The rest of the text, and the tool-call markup if any. */
  function flush() {
    const rest = held
    held = ""
    return { text: rest, markup }
  }

  return { push, flush }
}

/** A value written the way Python (and LFM2.5) writes it. */
function pythonValue(value: unknown): string {
  if (value === true) return "True"
  if (value === false) return "False"
  if (value === null || value === undefined) return "None"
  return JSON.stringify(value)
}

/** A past tool call in LFM2.5's own format, for the assistant turn's text. */
export function formatLfm2ToolCalls(calls: Array<ParsedToolCall>) {
  const list = calls
    .map(
      (c) =>
        `${c.name}(${Object.entries(c.args)
          .map(([k, v]) => `${k}=${pythonValue(v)}`)
          .join(", ")})`
    )
    .join(", ")
  return `${TOOL_CALL_START}[${list}]${TOOL_CALL_END}`
}
