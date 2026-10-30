/**
 * Splits a model's reply into its thinking and its answer. Thinking is any
 * `<think>…</think>` block, or everything before a lone `</think>` (models
 * prompted with an empty `<think></think>` sometimes still think a little and
 * close it). A `<think>` with no `</think>` yet means it's still thinking.
 */
export function splitThinking(text: string) {
  const thoughts: Array<string> = []
  let rest = text
  let isThinking = false
  for (;;) {
    const open = rest.indexOf("<think>")
    const close = rest.indexOf("</think>")
    if (close !== -1 && (open === -1 || close < open)) {
      // A close with no open: everything before it was thinking.
      thoughts.push(rest.slice(0, close))
      rest = rest.slice(close + "</think>".length)
      continue
    }
    if (open === -1) break
    const before = rest.slice(0, open)
    const after = rest.slice(open + "<think>".length)
    const end = after.indexOf("</think>")
    if (end === -1) {
      thoughts.push(after)
      rest = before
      isThinking = true
      break
    }
    thoughts.push(after.slice(0, end))
    rest = before + after.slice(end + "</think>".length)
  }
  return {
    thinking: thoughts
      .map((t) => t.trim())
      .filter(Boolean)
      .join("\n\n"),
    answer: rest.trim(),
    isThinking
  }
}

const OPEN = "<think>"
const CLOSE = "</think>"

/** Length of the longest end of `text` that could be the start of `tag`. */
function partialTag(text: string, tag: string) {
  for (let n = Math.min(tag.length - 1, text.length); n > 0; n--) {
    if (tag.startsWith(text.slice(-n))) return n
  }
  return 0
}

export type ReplyPiece = { kind: "thinking" | "answer"; text: string }

/**
 * Splits a reply into thinking and answer while it streams. A reply that
 * starts with `<think>` is thinking until `</think>`; the rest is the
 * answer. Tags split across tokens are held back until they are complete.
 */
export function createThinkingSplitter() {
  let state: "start" | "thinking" | "answer" = "start"
  let buffer = ""
  /** Whether the answer has had any text yet (to drop leading blank lines). */
  let answered = false

  const answer = (text: string): Array<ReplyPiece> => {
    const out = answered ? text : text.trimStart()
    if (!out) return []
    answered = true
    return [{ kind: "answer", text: out }]
  }

  function push(text: string): Array<ReplyPiece> {
    buffer += text
    const pieces: Array<ReplyPiece> = []
    for (;;) {
      if (state === "start") {
        const lead = buffer.trimStart()
        if (lead.startsWith(OPEN)) {
          buffer = lead.slice(OPEN.length)
          state = "thinking"
        } else if (OPEN.startsWith(lead)) {
          return pieces
        } else {
          state = "answer"
        }
      } else if (state === "thinking") {
        const end = buffer.indexOf(CLOSE)
        if (end === -1) {
          const keep = partialTag(buffer, CLOSE)
          const done = buffer.slice(0, buffer.length - keep)
          if (done) pieces.push({ kind: "thinking", text: done })
          buffer = buffer.slice(done.length)
          return pieces
        }
        if (end > 0) {
          pieces.push({ kind: "thinking", text: buffer.slice(0, end) })
        }
        buffer = buffer.slice(end + CLOSE.length)
        state = "answer"
      } else {
        pieces.push(...answer(buffer))
        buffer = ""
        return pieces
      }
    }
  }

  /** Whatever is still held back, once the reply has ended. */
  function flush(): Array<ReplyPiece> {
    const rest = buffer
    buffer = ""
    if (!rest) return []
    return state === "thinking"
      ? [{ kind: "thinking", text: rest }]
      : answer(rest)
  }

  return { push, flush }
}
