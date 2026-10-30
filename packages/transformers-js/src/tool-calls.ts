/**
 * Reading and writing tool calls in the local models' own formats:
 *
 * - LFM2.5: `<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>`
 * - LFM3: `<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>`
 * - Gemma 4: `<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>`
 * - Apple's model, when apfel doesn't recognize its call and passes it on as
 *   text: `{"tool_calls": [{"function": "get_weather", "arguments": {…}}]}`
 */

export type ParsedToolCall = { name: string; args: Record<string, unknown> }

/** How a model writes tool calls. */
export type ToolFormat = "lfm2" | "lfm3" | "gemma4"

const MARKERS = ["<|tool_call_start|>", "<tool_call>", "<|tool_call>"]

/** Where Apple's model starts a tool call it writes as text. */
export const JSON_TOOL_CALLS_MARKER = '{"tool_calls"'

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

/** `<function=name><parameter=k>v</parameter></function>` → calls. */
function parseXmlish(body: string) {
  const calls: Array<ParsedToolCall> = []
  for (const f of body.matchAll(
    /<function=([^>\n]+)>([\s\S]*?)(?:<\/function>|$)/g
  )) {
    const args: Record<string, unknown> = {}
    for (const p of f[2].matchAll(
      /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g
    )) {
      args[p[1].trim()] = parseValue(p[2])
    }
    calls.push({ name: f[1].trim(), args })
  }
  return calls
}

/** Gemma 4's string delimiter: `<|"|>text<|"|>`. */
const GEMMA_QUOTE = '<|"|>'

/**
 * A value in Gemma 4's notation: `<|"|>text<|"|>` strings, bare numbers,
 * true/false/null, `{key:value}` objects with bare keys, and `[…]` arrays.
 * Returns the value and where it ends.
 */
function gemmaValue(text: string, start: number): [unknown, number] {
  let i = start
  while (/\s/.test(text[i] ?? "")) i++
  if (text.startsWith(GEMMA_QUOTE, i)) {
    const end = text.indexOf(GEMMA_QUOTE, i + GEMMA_QUOTE.length)
    const stop = end < 0 ? text.length : end
    return [
      text.slice(i + GEMMA_QUOTE.length, stop),
      end < 0 ? stop : stop + GEMMA_QUOTE.length
    ]
  }
  if (text[i] === "{") return gemmaObject(text, i)
  if (text[i] === "[") {
    const items: Array<unknown> = []
    i++
    for (;;) {
      while (/[\s,]/.test(text[i] ?? "")) i++
      if (i >= text.length || text[i] === "]") return [items, i + 1]
      const [item, next] = gemmaValue(text, i)
      items.push(item)
      if (next <= i) return [items, text.length]
      i = next
    }
  }
  const raw = /^[^,}\]]*/.exec(text.slice(i))?.[0] ?? ""
  return [parseValue(raw), i + raw.length]
}

/** `{key:value,…}` in Gemma 4's notation, from `text[start]` (a `{`). */
function gemmaObject(
  text: string,
  start: number
): [Record<string, unknown>, number] {
  const object: Record<string, unknown> = {}
  let i = start + 1
  for (;;) {
    while (/[\s,]/.test(text[i] ?? "")) i++
    if (i >= text.length || text[i] === "}") return [object, i + 1]
    let key: string
    if (text.startsWith(GEMMA_QUOTE, i)) {
      const end = text.indexOf(GEMMA_QUOTE, i + GEMMA_QUOTE.length)
      if (end < 0) return [object, text.length]
      key = text.slice(i + GEMMA_QUOTE.length, end)
      i = end + GEMMA_QUOTE.length
    } else {
      const bare = /^[^:{}\s]+/.exec(text.slice(i))?.[0]
      if (!bare) return [object, text.length]
      key = bare
      i += bare.length
    }
    while (/\s/.test(text[i] ?? "")) i++
    if (text[i] !== ":") return [object, text.length]
    const [value, next] = gemmaValue(text, i + 1)
    object[key] = value
    i = next
  }
}

/** `call:name{…}` bodies of Gemma 4's `<|tool_call>…<tool_call|>` blocks. */
function parseGemma(text: string) {
  const calls: Array<ParsedToolCall> = []
  for (const m of text.matchAll(
    /<\|tool_call>\s*call:([^{\s]+)\s*([\s\S]*?)(?:<tool_call\|>|$)/g
  )) {
    const body = m[2].trim()
    const args = body.startsWith("{") ? gemmaObject(body, 0)[0] : {}
    calls.push({ name: m[1], args })
  }
  return calls
}

/** Every tool call in `text`, whichever format the model used. */
export function parseToolCalls(text: string): Array<ParsedToolCall> {
  const calls: Array<ParsedToolCall> = [...parseGemma(text)]
  for (const m of text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)) {
    const body = m[1].trim()
    if (body.includes("<function=")) {
      calls.push(...parseXmlish(body))
      continue
    }
    try {
      const json = JSON.parse(body) as {
        name: string
        arguments?: unknown
        parameters?: unknown
      }
      const args =
        typeof json.arguments === "string"
          ? JSON.parse(json.arguments)
          : (json.arguments ?? json.parameters ?? {})
      calls.push({ name: json.name, args: args as Record<string, unknown> })
    } catch {
      calls.push(...parsePythonic(body))
    }
  }
  for (const m of text.matchAll(
    /<\|tool_call_start\|>([\s\S]*?)(?:<\|tool_call_end\|>|$)/g
  )) {
    calls.push(...parsePythonic(m[1]))
  }
  return calls
}

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
export function createToolCallSplitter(markers: Array<string> = MARKERS) {
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
  return `<|tool_call_start|>[${list}]<|tool_call_end|>`
}
