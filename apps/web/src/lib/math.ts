/**
 * Prepares model output for remark-math, which only understands `$…$` and
 * `$$…$$`. Code is left untouched.
 *
 * - `\[ … \]` becomes a `$$` block and `\( … \)` becomes `$…$`: models use
 *   both LaTeX styles.
 * - A dollar amount like "$1.10 total" is escaped so it isn't read as the
 *   start of inline math. A number right before a closing `$` (as in `$0.05$`)
 *   still counts as math.
 * - A bare `\boxed{5}` (reasoning models' final-answer style) on a line with
 *   no other math is wrapped in `$…$` so it renders.
 */
export function normalizeMath(text: string) {
  // Split out fenced and inline code so math rules never touch it.
  return text
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) => (i % 2 === 1 ? part : normalizeProse(part)))
    .join("")
}

function normalizeProse(text: string) {
  return (
    text
      .replace(
        /\\\[([\s\S]+?)\\\]/g,
        (_, math: string) => `\n$$\n${math.trim()}\n$$\n`
      )
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, math: string) => `$${math.trim()}$`)
      // A price: "$" + a whole number (1, 1.10, 12,000) not followed by more
      // digits, a letter or a closing "$".
      .replace(
        /(^|[\s(])\$(\d[\d,]*(?:\.\d+)?)(?![\d$A-Za-z])(?![.,]\d)/gm,
        "$1\\$$$2"
      )
      // A bare \boxed{…} on a line with no other math.
      .replace(/^[^$\n]*\\boxed\{[^{}\n]*\}[^$\n]*$/gm, (line) =>
        line.replace(/\\boxed\{[^{}]*\}/g, (box) => `$${box}$`)
      )
  )
}
