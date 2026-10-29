/**
 * Code highlighting with @tanstack/highlight: only the languages registered
 * here are bundled (unknown ones fall back to plain text), and the GitHub
 * light and dark themes follow the app's theme.
 */
import { createHighlighter } from "@tanstack/highlight/core"
import { c } from "@tanstack/highlight/languages/c"
import { cpp } from "@tanstack/highlight/languages/cpp"
import { css } from "@tanstack/highlight/languages/css"
import { diff } from "@tanstack/highlight/languages/diff"
import { go } from "@tanstack/highlight/languages/go"
import { html } from "@tanstack/highlight/languages/html"
import { java } from "@tanstack/highlight/languages/java"
import { js } from "@tanstack/highlight/languages/js"
import { json } from "@tanstack/highlight/languages/json"
import { jsx } from "@tanstack/highlight/languages/jsx"
import { kotlin } from "@tanstack/highlight/languages/kotlin"
import { markdown } from "@tanstack/highlight/languages/markdown"
import { python } from "@tanstack/highlight/languages/python"
import { rust } from "@tanstack/highlight/languages/rust"
import { shell } from "@tanstack/highlight/languages/shell"
import { sql } from "@tanstack/highlight/languages/sql"
import { swift } from "@tanstack/highlight/languages/swift"
import { ts } from "@tanstack/highlight/languages/ts"
import { tsx } from "@tanstack/highlight/languages/tsx"
import { yaml } from "@tanstack/highlight/languages/yaml"
import { createTanStackMarkdownHighlighter } from "@tanstack/highlight/markdown"
import { createThemeCss } from "@tanstack/highlight/theme"
import { githubDarkTheme } from "@tanstack/highlight/themes/github-dark"
import { githubLightTheme } from "@tanstack/highlight/themes/github-light"

const highlighter = createHighlighter({
  languages: [
    c,
    cpp,
    css,
    diff,
    go,
    html,
    java,
    js,
    json,
    jsx,
    kotlin,
    markdown,
    python,
    rust,
    shell,
    sql,
    swift,
    ts,
    tsx,
    yaml
  ]
})

/** For TanStack Markdown's `highlighter` option. */
export const highlightMarkdownCode =
  createTanStackMarkdownHighlighter(highlighter)

/** Token colors for code blocks inside `.markdown`, light and dark. */
export const highlightThemeCss = createThemeCss({
  light: githubLightTheme,
  dark: githubDarkTheme,
  lightSelector: ".markdown",
  darkSelector: ".dark .markdown",
  codeBlockSelector: ".markdown pre.tm-code"
})
