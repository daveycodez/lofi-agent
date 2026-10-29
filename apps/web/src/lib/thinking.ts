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
