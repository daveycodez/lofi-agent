/**
 * Sources from tool results. Search results and DeepWiki answers are
 * rewritten into a short form for the model: numbered `[n] Title — URL`
 * lines it can cite. The same lines are read back to show the sources.
 */

export type Source = {
  /** The number the model sees, `[n]`. */
  n: number
  url: string
  title: string
  /** The site's host, without "www.". */
  site: string
}

/** Results kept from one search, and snippet length per result. */
const MAX_RESULTS = 8
const MAX_SNIPPET = 400

const LINE = /^\[(\d+)\] (.+?) — (https?:\/\/\S+)\s*$/gm

export function siteOf(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "")
  } catch {
    return url
  }
}

/** Collapses whitespace and Markdown noise into a short excerpt. */
function snippet(text: string) {
  const flat = text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^#+\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim()
  return flat.length > MAX_SNIPPET ? `${flat.slice(0, MAX_SNIPPET)}…` : flat
}

type Item = { title: string; url: string; text?: string }

function numbered(items: Array<Item>) {
  return items
    .slice(0, MAX_RESULTS)
    .map((item, i) => {
      const title = item.title.replace(/\s+/g, " ").trim() || siteOf(item.url)
      const line = `[${i + 1}] ${title} — ${item.url}`
      const text = item.text ? snippet(item.text) : ""
      return text ? `${line}\n${text}` : line
    })
    .join("\n\n")
}

/** Exa's text format: blocks of "Title: …\nURL: …\n…\nHighlights: …". */
function exaItems(text: string): Array<Item> {
  return text
    .split(/\n-{3,}\n/)
    .map((block) => {
      const title = /^Title: (.*)$/m.exec(block)?.[1] ?? ""
      const url = /^URL: (\S+)$/m.exec(block)?.[1] ?? ""
      const body = block.split(/^Highlights:\s*$/m)[1] ?? ""
      return { title, url, text: body }
    })
    .filter((item) => /^https?:\/\//.test(item.url))
}

/** JSON results: `results[]` or `data.web[]` with url and title. */
function jsonItems(text: string): Array<Item> {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    return []
  }
  const root = data as {
    results?: Array<Record<string, unknown>>
    data?: { web?: Array<Record<string, unknown>> }
  }
  const list = root.results ?? root.data?.web ?? []
  return list
    .map((r) => ({
      title: String(r.title ?? ""),
      url: String(r.url ?? ""),
      text: [
        r.description,
        r.snippet,
        r.text,
        Array.isArray(r.excerpts) ? r.excerpts.join(" ") : undefined
      ]
        .filter((t) => typeof t === "string")
        .join(" ")
    }))
    .filter((item) => /^https?:\/\//.test(item.url))
}

/**
 * DeepWiki answers: `<cite repo path start end/>` tags point at source
 * files, and the answer ends with wiki page links and a link to the search.
 */
function deepWikiAnswer(text: string) {
  const items: Array<Item> = []
  const seen = new Set<string>()
  const add = (item: Item) => {
    if (seen.has(item.url)) return
    seen.add(item.url)
    items.push(item)
  }
  for (const m of text.matchAll(/<cite\s+([^>]*?)\/?>/g)) {
    const attr = (name: string) =>
      new RegExp(`${name}="([^"]*)"`).exec(m[1])?.[1] ?? ""
    const repo = attr("repo")
    const path = attr("path")
    if (!repo) continue
    if (!path || path === "wiki_page") {
      add({ title: `${repo} wiki`, url: `https://deepwiki.com/${repo}` })
      continue
    }
    const start = attr("start")
    const end = attr("end")
    const lines = /^\d+$/.test(start)
      ? `#L${start}${/^\d+$/.test(end) && end !== start ? `-L${end}` : ""}`
      : ""
    add({
      title: `${path.split("/").at(-1)} (${repo})`,
      url: `https://github.com/${repo}/blob/HEAD/${path}${lines}`
    })
  }
  for (const m of text.matchAll(
    /\[([^\]]+)\]\(\/wiki\/([^/)#\s]+\/[^/)#\s]+)/g
  )) {
    add({ title: m[1], url: `https://deepwiki.com/${m[2]}` })
  }
  const search = /View this search on DeepWiki: (https:\/\/\S+)/.exec(text)
  if (search) add({ title: "DeepWiki search", url: search[1] })
  if (items.length === 0) return null
  const body = text
    .replace(/\s*<cite\s+[^>]*?\/?>/g, "")
    .replace(/\n*Wiki pages you might want to explore:[\s\S]*$/, "")
    .trim()
  return `${body}\n\nSources:\n${numbered(items)}`
}

/**
 * A tool result in the short form above, when it holds search results or a
 * DeepWiki answer. Anything else comes back unchanged.
 */
export function compactToolResult(text: string) {
  const items = jsonItems(text)
  if (items.length > 0) return numbered(items)
  const exa = exaItems(text)
  if (exa.length > 0) return numbered(exa)
  return deepWikiAnswer(text) ?? text
}

/** The numbered sources in a tool result written by compactToolResult. */
export function parseSources(text: string): Array<Source> {
  return [...text.matchAll(LINE)].map((m) => ({
    n: Number(m[1]),
    title: m[2],
    url: m[3],
    site: siteOf(m[3])
  }))
}
