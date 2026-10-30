/**
 * The built-in web search tool. The browser calls each search provider's
 * keyless MCP endpoint directly, one search per call, trying them in order
 * and moving on when one is rate-limited, fails or finds nothing. Results
 * come back as numbered sources the chat shows as chips.
 */
import { toolDefinition } from "@tanstack/ai"
import { useMemo } from "react"

import { webSearchEnabledStore } from "#/lib/settings"
import { compactToolResult, parseSources, shorten } from "#/lib/sources"

/** What the model asks for: what to search, and what it wants to find out. */
type Search = { query: string; objective: string }

type Provider = {
  name: string
  url: string
  tool: string
  /** The provider's own arguments for a search. */
  args: (search: Search) => Record<string, unknown>
}

/**
 * Best results first. Exa and Parallel return excerpts picked for the
 * query; Firecrawl only search-page snippets.
 */
const PROVIDERS: Array<Provider> = [
  {
    name: "Exa",
    url: "https://mcp.exa.ai/mcp",
    tool: "web_search_exa",
    args: ({ query, objective }) => ({ query, objective, numResults: 5 })
  },
  {
    name: "Parallel",
    url: "https://search.parallel.ai/mcp",
    tool: "web_search",
    args: ({ query, objective }) => ({ objective, search_queries: [query] })
  },
  {
    name: "Firecrawl",
    url: "https://mcp.firecrawl.dev/v2/mcp",
    tool: "firecrawl_search",
    args: ({ query, objective }) => ({ query, objective, limit: 5 })
  }
]

/** A provider that rate-limited this browser is skipped for this long. */
const RATE_LIMIT_PAUSE_MS = 10 * 60 * 1000
/** A provider that hasn't answered by then is skipped for this search. */
const TIMEOUT_MS = 15_000

/** When each rate-limited provider can be tried again. */
const pausedUntil = new Map<string, number>()

class RateLimited extends Error {}

const isRateLimit = (text: string) =>
  /rate.?limit|too many requests|\b429\b|quota/i.test(text)

/** One search with one provider: the result text, as the model will see it. */
async function search(provider: Provider, request: Search) {
  const res = await fetch(provider.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream"
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: { name: provider.tool, arguments: provider.args(request) }
    }),
    signal: AbortSignal.timeout(TIMEOUT_MS)
  })
  const body = await res.text()
  if (res.status === 429 || isRateLimit(res.statusText)) {
    throw new RateLimited(`HTTP ${res.status}`)
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  // Answered as JSON, or as an event stream with the JSON on a data: line.
  const json =
    body
      .split("\n")
      .find((line) => line.startsWith("data:"))
      ?.slice(5) ?? body
  const reply = JSON.parse(json) as {
    error?: { message?: string }
    result?: {
      isError?: boolean
      content?: Array<{ type: string; text?: string }>
    }
  }
  const text = reply.result?.content?.map((c) => c.text ?? "").join("\n") ?? ""
  const error = reply.error?.message ?? (reply.result?.isError ? text : "")
  if (error) {
    throw isRateLimit(error) ? new RateLimited(error) : new Error(error)
  }
  const compact = compactToolResult(text)
  if (parseSources(compact).length === 0) throw new Error("No results.")
  return shorten(compact)
}

/** Searches with the first provider that answers. */
async function webSearch(input: unknown) {
  const request = input as Search
  const failures: Array<string> = []
  for (const provider of PROVIDERS) {
    if ((pausedUntil.get(provider.name) ?? 0) > Date.now()) {
      failures.push(`${provider.name}: rate-limited`)
      continue
    }
    try {
      return await search(provider, request)
    } catch (error) {
      if (error instanceof RateLimited) {
        pausedUntil.set(provider.name, Date.now() + RATE_LIMIT_PAUSE_MS)
      }
      failures.push(
        `${provider.name}: ${error instanceof Error ? error.message : String(error)}`
      )
    }
  }
  throw new Error(
    `Web search is unavailable right now (${failures.join("; ")}).`
  )
}

const webSearchTool = toolDefinition({
  name: "web_search",
  description: "Search the web for current information.",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "What to search for." },
      objective: {
        type: "string",
        description: "What you want to find out from the results."
      }
    },
    required: ["query", "objective"],
    additionalProperties: false
  }
}).client(webSearch)

/** The web search tool when it's on in Settings, else none. */
export function useWebSearchTools() {
  const enabled = webSearchEnabledStore.use()
  return useMemo(() => (enabled ? [webSearchTool] : []), [enabled])
}
