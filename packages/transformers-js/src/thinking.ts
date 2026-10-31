import type { ToolFormat } from "./tool-calls.js"

/** Where a model's thinking starts and ends in its reply. */
export type ThinkingMarkers = { open: string; close: string }

/** `<think>…</think>`, as LFM models (and most others) write it. */
export const THINK_TAGS: ThinkingMarkers = {
  open: "<think>",
  close: "</think>"
}

/** Each format's thinking markers. */
export const THINKING_MARKERS: Record<ToolFormat, ThinkingMarkers> = {
  lfm2: THINK_TAGS,
  lfm3: THINK_TAGS,
  // Gemma 4 thinks in a "thought" channel: `<|channel>thought\n…<channel|>`.
  gemma4: { open: "<|channel>thought", close: "<channel|>" }
}

/**
 * Splits a model's reply into its thinking and its answer. Thinking is any
 * `<think>…</think>` block (or the format's own markers), or everything
 * before a lone close (models prompted with an empty `<think></think>`
 * sometimes still think a little and close it). An open with no close yet
 * means it's still thinking.
 */
export function splitThinking(
  text: string,
  { open: OPEN, close: CLOSE }: ThinkingMarkers = THINK_TAGS
) {
  const thoughts: Array<string> = []
  let rest = text
  let isThinking = false
  for (;;) {
    const open = rest.indexOf(OPEN)
    const close = rest.indexOf(CLOSE)
    if (close !== -1 && (open === -1 || close < open)) {
      // A close with no open: everything before it was thinking.
      thoughts.push(rest.slice(0, close))
      rest = rest.slice(close + CLOSE.length)
      continue
    }
    if (open === -1) break
    const before = rest.slice(0, open)
    const after = rest.slice(open + OPEN.length)
    const end = after.indexOf(CLOSE)
    if (end === -1) {
      thoughts.push(after)
      rest = before
      isThinking = true
      break
    }
    thoughts.push(after.slice(0, end))
    rest = before + after.slice(end + CLOSE.length)
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
 * starts with the open marker (`<think>`) is thinking until the close
 * (`</think>`); the rest is the answer, with any stray close left out.
 * Markers split across tokens are held back until they are complete.
 */
export function createThinkingSplitter({
  open: OPEN,
  close: CLOSE
}: ThinkingMarkers = THINK_TAGS) {
  let state: "start" | "thinking" | "answer" = "start"
  let buffer = ""
  /** Whether the thinking has had any text yet (to drop a leading newline). */
  let thought = false
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
          buffer = buffer.slice(done.length)
          const text = thought ? done : done.trimStart()
          if (text) {
            thought = true
            pieces.push({ kind: "thinking", text })
          }
          return pieces
        }
        const text = thought
          ? buffer.slice(0, end)
          : buffer.slice(0, end).trimStart()
        if (text) {
          thought = true
          pieces.push({ kind: "thinking", text })
        }
        buffer = buffer.slice(end + CLOSE.length)
        state = "answer"
      } else {
        // A close in the answer is dropped: a model told not to think
        // sometimes still thinks a little and closes it, and what it wrote
        // has already streamed as the answer.
        buffer = buffer.split(CLOSE).join("")
        const keep = partialTag(buffer, CLOSE)
        pieces.push(...answer(buffer.slice(0, buffer.length - keep)))
        buffer = buffer.slice(buffer.length - keep)
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
