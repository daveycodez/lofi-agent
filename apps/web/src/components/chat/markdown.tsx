/**
 * Renders model output as Markdown, with GitHub tables and lists, LaTeX math
 * (KaTeX) and code highlighting. Raw HTML in the text is not rendered.
 */
import "katex/dist/katex.min.css"
import bash from "highlight.js/lib/languages/bash"
import c from "highlight.js/lib/languages/c"
import cpp from "highlight.js/lib/languages/cpp"
import css from "highlight.js/lib/languages/css"
import diff from "highlight.js/lib/languages/diff"
import go from "highlight.js/lib/languages/go"
import java from "highlight.js/lib/languages/java"
import javascript from "highlight.js/lib/languages/javascript"
import json from "highlight.js/lib/languages/json"
import kotlin from "highlight.js/lib/languages/kotlin"
import markdown from "highlight.js/lib/languages/markdown"
import python from "highlight.js/lib/languages/python"
import rust from "highlight.js/lib/languages/rust"
import sql from "highlight.js/lib/languages/sql"
import swift from "highlight.js/lib/languages/swift"
import typescript from "highlight.js/lib/languages/typescript"
import xml from "highlight.js/lib/languages/xml"
import yaml from "highlight.js/lib/languages/yaml"
import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import rehypeKatex from "rehype-katex"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"

import { normalizeMath } from "#/lib/math"

/**
 * Highlighting: a small set of common languages (their aliases like js, ts,
 * sh and html come with them), and only for code blocks that name their
 * language. Auto-detection would try every language on every block.
 */
const LANGUAGES = {
  bash,
  c,
  cpp,
  css,
  diff,
  go,
  java,
  javascript,
  json,
  kotlin,
  markdown,
  python,
  rust,
  sql,
  swift,
  typescript,
  xml,
  yaml
}

type MdNode = {
  type: string
  value?: string
  children?: Array<MdNode>
  data?: Record<string, unknown>
}

const displayMath = (value = ""): MdNode => ({
  type: "math",
  value,
  data: {
    hName: "pre",
    hChildren: [
      {
        type: "element",
        tagName: "code",
        properties: { className: ["language-math", "math-display"] },
        children: [{ type: "text", value }]
      }
    ]
  }
})

/** Splits a paragraph's children into lines, at hard breaks and newlines. */
function splitLines(children: Array<MdNode>) {
  const lines: Array<Array<MdNode>> = [[]]
  for (const child of children) {
    if (child.type === "break") {
      lines.push([])
    } else if (child.type === "text" && child.value?.includes("\n")) {
      child.value.split("\n").forEach((piece, j) => {
        if (j > 0) lines.push([])
        if (piece) lines.at(-1)?.push({ type: "text", value: piece })
      })
    } else {
      lines.at(-1)?.push(child)
    }
  }
  return lines
}

const isBlank = (n: MdNode) => n.type === "text" && !n.value?.trim()

/**
 * Models often write equations as `$…$` alone on their own lines. Show each
 * line made only of an equation as display math (centered, a touch larger),
 * and keep the paragraph's other lines as text. Only top-level paragraphs:
 * equations in list items stay inline next to their numbers.
 */
function remarkLoneMathToDisplay() {
  const visit = (node: MdNode) => {
    const list = node.children ?? []
    for (let i = 0; i < list.length; i++) {
      const child = list[i]
      // Top-level paragraphs only: equations in lists stay inline.
      if (child.type !== "paragraph" || !child.children) continue
      const lines = splitLines(child.children).map((l) =>
        l.filter((n) => !isBlank(n))
      )
      const isMath = (l: Array<MdNode>) =>
        l.length === 1 && l[0].type === "inlineMath"
      if (!lines.some(isMath)) continue
      // Rebuild: equation lines become display math, runs of other lines stay
      // a paragraph (joined with line breaks).
      const blocks: Array<MdNode> = []
      let text: Array<MdNode> = []
      const flush = () => {
        if (text.length) blocks.push({ type: "paragraph", children: text })
        text = []
      }
      for (const line of lines) {
        if (isMath(line)) {
          flush()
          blocks.push(displayMath(line[0].value))
        } else if (line.length) {
          if (text.length) text.push({ type: "break" })
          text.push(...line)
        }
      }
      flush()
      list.splice(i, 1, ...blocks)
      i += blocks.length - 1
    }
  }
  return (tree: MdNode) => visit(tree)
}

type HastNode = {
  type: string
  value?: string
  tagName?: string
  properties?: { className?: Array<string> }
  children?: Array<HastNode>
}

/**
 * Keeps inline math together with the punctuation right after it, so a line
 * never starts with a lone "." or ",".
 */
function rehypeMathPunctuation() {
  const visit = (node: HastNode) => {
    const list = node.children ?? []
    for (let i = 0; i < list.length; i++) {
      const child = list[i]
      const next = list[i + 1]
      const isInlineMath =
        child.type === "element" &&
        child.properties?.className?.includes("katex")
      const punct =
        next?.type === "text" ? /^[.,;:!?)]+/.exec(next.value ?? "") : null
      if (isInlineMath && next && punct) {
        list[i] = {
          type: "element",
          tagName: "span",
          properties: { className: ["math-nowrap"] },
          children: [child, { type: "text", value: punct[0] }]
        }
        next.value = (next.value ?? "").slice(punct[0].length)
      } else if (!isInlineMath) {
        visit(child)
      }
    }
  }
  return (tree: HastNode) => visit(tree)
}

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath, remarkLoneMathToDisplay]}
      rehypePlugins={[
        [rehypeKatex, { throwOnError: false }],
        rehypeMathPunctuation,
        [rehypeHighlight, { languages: LANGUAGES, detect: false }]
      ]}
      components={{
        a: ({ node: _node, children: label, ...props }) => (
          <a {...props} target="_blank" rel="noopener noreferrer">
            {label}
          </a>
        )
      }}
    >
      {normalizeMath(children)}
    </ReactMarkdown>
  )
}
