/**
 * Prepares model output for tanstack-markdown-math, which only understands
 * `$…$` and `$$…$$`. Code is left untouched.
 *
 * - `\[ … \]` becomes a `$$` block and `\( … \)` becomes `$…$`: models use
 *   both LaTeX styles.
 * - An equation alone on its own line (`$x = 5$`) becomes a `$$` block, so it
 *   shows centered like display math. Indented lines and list items stay
 *   inline next to their text.
 * - A dollar amount like "$1.10 total" is escaped so it isn't read as the
 *   start of inline math. It stays math when a later `$` on the line closes
 *   an equation (`$3 \times 6 = 18$`), or right before a closing `$` (`$0.05$`).
 * - A bare `\boxed{5}` (reasoning models' final-answer style) on a line with
 *   no other math is wrapped in `$…$` so it renders.
 * - Punctuation right after inline math moves inside it, so a line never
 *   starts with a lone "." or ",".
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
        (_, math: string) => `\n\n$$\n${math.trim()}\n$$\n\n`
      )
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, math: string) => `$${math.trim()}$`)
      // An equation alone on a line.
      .replace(
        /^\$(?![\s$])([^$\n]*?[^\s\\$])\$[ \t]*$/gm,
        (_, math: string) => `\n$$\n${math}\n$$\n`
      )
      // A price: "$" + a number (1, 1.10, 12,000) not followed by more
      // digits or a letter, unless a "$" later on the line closes an
      // equation ("$3 \times 6 = 18$").
      .replace(
        /(^|[\s(])\$(\d[\d,]*(?:\.\d+)?)(?![\d$A-Za-z])(?![.,]\d)(?=([^\n]*))/gm,
        (match, before: string, amount: string, rest: string) => {
          const closed = /^([^$]*)\$(?!\d)/.exec(rest)
          return closed && /[\\=^_{}]/.test(closed[1])
            ? match
            : `${before}\\$${amount}`
        }
      )
      // A bare \boxed{…} on a line with no other math.
      .replace(/^[^$\n]*\\boxed\{[^{}\n]*\}[^$\n]*$/gm, (line) =>
        line.replace(/\\boxed\{[^{}]*\}/g, (box) => `$${box}$`)
      )
      // Inline math, then punctuation: "$x$." becomes "$x\text{.}$".
      .replace(
        /(?<![\\$])\$(?![\s$])([^$\n]*?[^\s\\$])\$(?![\d$])([.,;:!?)]+)/g,
        (_, math: string, punct: string) => `$${math}\\text{${punct}}$`
      )
  )
}
