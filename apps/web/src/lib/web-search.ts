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

/** What the model asks for: the query, and what it wants from the results. */
type Search = { query: string; objective: string }

type Provider = {
  name: string
  url: string
  tool: string
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
  const { query, objective } = (input ?? {}) as Partial<Search>
  if (typeof query !== "string" || !query.trim()) {
    throw new Error("The search query is empty.")
  }
  const request = {
    query,
    objective:
      typeof objective === "string" && objective.trim() ? objective : query
  }
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
  throw new Error(`Web search is unavailable right now (${failures.join("; ")}).`)
}

/*
 * Exa's own search tool definition, as its MCP server lists it, so models
 * see what they did when Exa was an MCP server. Left out: the sentence about
 * its page-fetch tool, and its category:people / category:company search
 * syntax, which only Exa understands and which Apple's model wrote into its
 * arguments as invalid JSON.
 */
const webSearchTool = toolDefinition({
  name: "web_search",
  description:
    'Search the web for any topic and get clean, ready-to-use content.\n\n      Best for: Finding current information, news, facts, people, companies, or answering questions about any topic.\n      Returns: Clean text content from top search results.\n\n      Query tips:\n      describe the ideal page, not keywords. "blog post comparing React and Vue performance" not "React vs Vue".',
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        minLength: 1,
        description:
          "Natural language search query. Should be a semantically rich description of the ideal page, not just keywords."
      },
      numResults: {
        type: "number",
        description: "Number of search results to return (default: 10)."
      },
      objective: {
        type: "string",
        minLength: 1,
        maxLength: 4096,
        description:
          "Goal for this search turn; say which documents should rank first, which should be excluded, and what specific facts or figures to pull from them."
      }
    },
    required: ["query", "objective"],
    additionalProperties: false,
    $schema: "http://json-schema.org/draft-07/schema#"
  }
}).client(webSearch)

/** The web search tool when it's on in Settings, else none. */
export function useWebSearchTools() {
  const enabled = webSearchEnabledStore.use()
  return useMemo(() => (enabled ? [webSearchTool] : []), [enabled])
}
