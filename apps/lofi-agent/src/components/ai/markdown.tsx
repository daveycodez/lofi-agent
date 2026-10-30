/**
 * Renders model output with TanStack Markdown, set up like TanStack AI's
 * `TextPart`: the streaming extension (unfinished syntax renders cleanly
 * while text arrives), no frontmatter or heading ids, and raw HTML shown as
 * text. Math comes from tanstack-markdown-math (KaTeX) and code highlighting
 * from @tanstack/highlight with the GitHub themes.
 */
import "katex/dist/katex.min.css"
import Check from "@gravity-ui/icons/Check"
import Copy from "@gravity-ui/icons/Copy"
import { Button, Tooltip } from "@heroui/react"
import { Markdown as TanStackMarkdown } from "@tanstack/markdown/react"
import type { MarkdownComponents } from "@tanstack/markdown/react"
import { streamingMarkdownExtension } from "@tanstack/markdown/extensions/streaming"
import { useEffect, useMemo, useRef, useState } from "react"
import type { ComponentPropsWithoutRef } from "react"
import { mathExtension } from "tanstack-markdown-math"
import { MathBlock, MathInline } from "tanstack-markdown-math/react"

import { CitationChip } from "#/components/ai/sources"
import { highlightMarkdownCode, highlightThemeCss } from "#/lib/highlight"
import { normalizeMath } from "#/lib/math"
import type { Source } from "#/lib/sources"

const extensions = [
  streamingMarkdownExtension(),
  ...mathExtension({ output: "component" })
]

/** Readable names for the languages a code block's header shows. */
const LANGUAGE_NAMES: Record<string, string> = {
  bash: "Bash",
  sh: "Shell",
  shell: "Shell",
  zsh: "Shell",
  c: "C",
  cpp: "C++",
  css: "CSS",
  diff: "Diff",
  go: "Go",
  html: "HTML",
  java: "Java",
  javascript: "JavaScript",
  js: "JavaScript",
  jsx: "JSX",
  json: "JSON",
  kotlin: "Kotlin",
  markdown: "Markdown",
  md: "Markdown",
  python: "Python",
  py: "Python",
  rust: "Rust",
  sql: "SQL",
  swift: "Swift",
  ts: "TypeScript",
  tsx: "TSX",
  typescript: "TypeScript",
  xml: "XML",
  yaml: "YAML",
  yml: "YAML"
}

function CopyCodeButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const t = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(t)
  }, [copied])
  return (
    <Tooltip delay={300}>
      <Button
        isIconOnly
        size="sm"
        variant="ghost"
        aria-label={copied ? "Copied" : "Copy code"}
        className="text-muted"
        onPress={() => {
          void navigator.clipboard.writeText(getText())
          setCopied(true)
        }}
      >
        {copied ? <Check /> : <Copy />}
      </Button>
      <Tooltip.Content>{copied ? "Copied" : "Copy code"}</Tooltip.Content>
    </Tooltip>
  )
}

/** A fenced code block: a header with its language and a copy button. */
function CodeBlock(props: ComponentPropsWithoutRef<"pre">) {
  const pre = useRef<HTMLPreElement>(null)
  const language = (props as { "data-lang"?: string })["data-lang"]
  const label =
    language && language !== "plaintext"
      ? (LANGUAGE_NAMES[language] ?? language)
      : "Code"
  return (
    <div className="code-block not-prose my-3 overflow-hidden rounded-2xl bg-surface-secondary">
      <div className="flex h-9 items-center justify-between ps-3.5 pe-1 text-xs text-muted">
        <span>{label}</span>
        <CopyCodeButton
          getText={() => (pre.current?.textContent ?? "").replace(/\n$/, "")}
        />
      </div>
      <pre {...props} ref={pre} />
    </div>
  )
}

function Link({ children, ...props }: ComponentPropsWithoutRef<"a">) {
  return (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  )
}

/**
 * Turns `[1]` and `[1, 2]` citations of known sources into `#cite-n` links,
 * which render as source chips. Code is left alone.
 */
function linkCitations(text: string, sources: Array<Source>) {
  const known = new Set(sources.map((s) => s.n))
  return text
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(
            /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?![(:])/g,
            (match, list: string) => {
              const ns = list.split(",").map((n) => Number(n.trim()))
              return ns.every((n) => known.has(n))
                ? ns.map((n) => `[${n}](#cite-${n})`).join("")
                : match
            }
          )
    )
    .join("")
}

const components: MarkdownComponents = {
  MathBlock,
  MathInline,
  pre: CodeBlock,
  a: Link
}

export function Markdown({
  children,
  sources = []
}: {
  children: string
  /** Sources the text may cite as `[n]`. */
  sources?: Array<Source>
}) {
  // Citation links render as chips for their source.
  const withCitations = useMemo<MarkdownComponents>(() => {
    if (sources.length === 0) return components
    const byN = new Map(sources.map((s) => [s.n, s]))
    return {
      ...components,
      a: (props: ComponentPropsWithoutRef<"a">) => {
        const n = /^#cite-(\d+)$/.exec(props.href ?? "")?.[1]
        const source = n ? byN.get(Number(n)) : undefined
        return source ? <CitationChip source={source} /> : <Link {...props} />
      }
    }
  }, [sources])
  return (
    <>
      {/* Code colors, added to the page once however many messages render. */}
      <style href="markdown-highlight" precedence="default">
        {highlightThemeCss}
      </style>
      <TanStackMarkdown
        extensions={extensions}
        frontmatter={false}
        headingIds={false}
        highlighter={highlightMarkdownCode}
        components={withCitations}
      >
        {normalizeMath(
          sources.length > 0 ? linkCitations(children, sources) : children
        )}
      </TanStackMarkdown>
    </>
  )
}
