/**
 * MCP servers: the built-in ones, then those the user adds. The browser
 * connects to each one directly with TanStack AI's MCP client (Streamable
 * HTTP), so a server only works if it allows requests from this site
 * (CORS). The list, on/off choices and any tokens stay in this browser's
 * storage.
 */
import { toolDefinition } from "@tanstack/ai"
import type { JSONSchema } from "@tanstack/ai"
import type { MCPClient } from "@tanstack/ai-mcp"
import { useMemo } from "react"
import { v7 as uuidv7 } from "uuid"

import { compactToolResult } from "#/lib/sources"
import { createStore } from "#/lib/store"

export type McpServer = {
  id: string
  name: string
  url: string
  /** Sent as `Authorization: Bearer <token>` when set. */
  token?: string
  enabled: boolean
  /**
   * Comes with the app: always listed first, and can't be removed. Its tools
   * are offered first too: small models tend to pick the first tool that
   * might fit, so the general ones should lead.
   */
  builtIn?: boolean
  /** Only these of the server's tools are used (all of them when unset). */
  onlyTools?: Array<string>
  /**
   * Only these of each tool's arguments are offered to the model (all of
   * them when unset).
   */
  onlyArgs?: Array<string>
  /** Arguments sent with every call, on top of the model's. */
  fixedArgs?: Record<string, unknown>
  /** Offer only the first paragraph of each tool's description. */
  shortDescription?: boolean
}

export type McpTool = {
  name: string
  description: string
  /** JSON Schema for the tool's arguments. */
  inputSchema: JSONSchema
}

export type McpServerStatus =
  | { state: "checking" }
  | { state: "connected"; tools: Array<McpTool> }
  | { state: "error"; message: string }

/** Servers that come with the app. They're on unless the user turns them off. */
const BUILT_IN_SERVERS: Array<Omit<McpServer, "enabled">> = [
  {
    // Web search. Works without a key (rate-limited per user).
    id: "built-in:firecrawl",
    name: "Firecrawl Web Search",
    url: "https://mcp.firecrawl.dev/v2/mcp",
    builtIn: true,
    // Search only: small models scrape made-up page URLs, and the search
    // results already carry the relevant text.
    onlyTools: ["firecrawl_search"],
    // The rest of the tool's description and arguments cover paid features
    // (data providers, scraping every result, enterprise modes). Offered to a
    // small model they make the prompt several times longer, and it fills
    // them in.
    onlyArgs: ["query", "objective"],
    shortDescription: true,
    // The app keeps 8 results at most; 5 keeps the prompt short.
    fixedArgs: { limit: 5 }
  }
]

/** On or off for each built-in server, by id. Missing means on. */
const builtInEnabledStore = createStore<Record<string, boolean>>(
  {},
  "lofi-agent:mcp-built-in"
)

/** Servers the user added. */
const customServersStore = createStore<Array<McpServer>>(
  [],
  "lofi-agent:mcp-servers"
)

/** A URL in one form, to spot the same server added twice. */
function sameUrl(url: string) {
  try {
    const u = new URL(url.trim())
    return `${u.origin}${u.pathname.replace(/\/+$/, "")}${u.search}`
  } catch {
    return url.trim()
  }
}

/** Built-in servers first, then the user's (minus copies of a built-in). */
function allServers(
  custom: Array<McpServer>,
  builtInEnabled: Record<string, boolean>
): Array<McpServer> {
  const builtIns = BUILT_IN_SERVERS.map((s) => ({
    ...s,
    enabled: builtInEnabled[s.id] ?? true
  }))
  const urls = new Set(builtIns.map((s) => sameUrl(s.url)))
  return [...builtIns, ...custom.filter((s) => !urls.has(sameUrl(s.url)))]
}

export const getMcpServers = () =>
  allServers(customServersStore.get(), builtInEnabledStore.get())

export function useMcpServers() {
  const custom = customServersStore.use()
  const builtInEnabled = builtInEnabledStore.use()
  return useMemo(
    () => allServers(custom, builtInEnabled),
    [custom, builtInEnabled]
  )
}

/** Whether a server with this URL is already in the list. */
export const hasMcpServer = (url: string) =>
  getMcpServers().some((s) => sameUrl(s.url) === sameUrl(url))

/** Connection status by server id, for this session only. */
export const mcpStatusStore = createStore<Record<string, McpServerStatus>>({})

const setStatus = (id: string, status: McpServerStatus) =>
  mcpStatusStore.set((all) => ({ ...all, [id]: status }))

/** A readable reason a connection failed. */
function describeError(error: unknown) {
  // The client wraps failures ("Failed to connect to MCP server"); the
  // reason is further down the cause chain.
  const messages: Array<string> = []
  for (let e = error, depth = 0; e != null && depth < 5; depth++) {
    messages.push(e instanceof Error ? e.message : String(e))
    e = e instanceof Error ? e.cause : undefined
  }
  const message = messages.join(" | ")
  // Chrome says "Failed to fetch" and Safari "Load failed" when a request is
  // blocked, which for a reachable server almost always means CORS.
  if (/failed to fetch|load failed|networkerror/i.test(message)) {
    return "Couldn't connect. The server may be offline, or it doesn't allow connections from this site."
  }
  if (/\b401\b|unauthori[sz]ed/i.test(message)) {
    return "The server asked for sign-in. Add its access token, or check the one you entered."
  }
  if (/\b403\b|forbidden/i.test(message)) {
    return "The server refused access with this token."
  }
  if (/\b404\b/.test(message)) {
    return "No MCP server at this URL. Check the path (often it ends in /mcp)."
  }
  return messages.at(-1) ?? message
}

/** Open connections by server id, reused for tool calls. */
const clients = new Map<string, Promise<MCPClient>>()

function connect(server: McpServer) {
  let client = clients.get(server.id)
  if (!client) {
    client = import("@tanstack/ai-mcp").then(({ createMCPClient }) =>
      createMCPClient({
        transport: {
          type: "http",
          url: server.url,
          ...(server.token && {
            headers: { Authorization: `Bearer ${server.token}` }
          })
        }
      })
    )
    clients.set(server.id, client)
    // A failed connection isn't kept, so the next try reconnects.
    client.catch(() => clients.delete(server.id))
  }
  return client
}

function disconnect(id: string) {
  const client = clients.get(id)
  clients.delete(id)
  void client?.then((c) => c.close()).catch(() => undefined)
}

/**
 * Removes every sentence that names one of `hidden` from a description, so a
 * model isn't told to use a tool it doesn't have.
 */
function withoutMentions(text: string, hidden: Array<string>) {
  if (hidden.length === 0) return text
  return text
    .split("\n")
    .map((line) =>
      line
        .split(/(?<=[.!?])\s+/)
        .filter((sentence) => !hidden.some((name) => sentence.includes(name)))
        .join(" ")
    )
    .filter((line, i, lines) => line.trim() || lines[i - 1]?.trim())
    .join("\n")
    .trim()
}

/** The same, for every description inside a JSON Schema. */
function schemaWithoutMentions(
  schema: unknown,
  hidden: Array<string>
): unknown {
  if (hidden.length === 0 || !schema || typeof schema !== "object") {
    return schema
  }
  if (Array.isArray(schema)) {
    return schema.map((item) => schemaWithoutMentions(item, hidden))
  }
  return Object.fromEntries(
    Object.entries(schema).map(([key, value]) => [
      key,
      key === "description" && typeof value === "string"
        ? withoutMentions(value, hidden)
        : schemaWithoutMentions(value, hidden)
    ])
  )
}

/** A tool's argument schema with only the `keep` arguments. */
function onlyArgs(schema: object, keep: Array<string> | undefined) {
  const s = schema as {
    properties?: Record<string, unknown>
    required?: Array<string>
  }
  if (!keep || !s.properties) return schema
  return {
    ...s,
    properties: Object.fromEntries(
      Object.entries(s.properties).filter(([name]) => keep.includes(name))
    ),
    ...(s.required && {
      required: s.required.filter((name) => keep.includes(name))
    })
  }
}

/** Connects to `server`, lists its tools, and records the result. */
export async function checkMcpServer(server: McpServer) {
  disconnect(server.id)
  setStatus(server.id, { state: "checking" })
  try {
    const all = await (await connect(server)).tools()
    const keep = (name: string) =>
      !server.onlyTools || server.onlyTools.includes(name)
    // Tools left out, which the kept tools' descriptions mustn't mention.
    const hidden = all.map((t) => t.name).filter((name) => !keep(name))
    setStatus(server.id, {
      state: "connected",
      tools: all
        .filter((t) => keep(t.name))
        .map((t) => ({
          name: t.name,
          description: withoutMentions(
            server.shortDescription
              ? t.description.trim().split(/\n\s*\n/)[0]
              : t.description,
            hidden
          ),
          inputSchema: schemaWithoutMentions(
            onlyArgs(t.inputSchema ?? { type: "object" }, server.onlyArgs),
            hidden
          ) as JSONSchema
        }))
    })
  } catch (error) {
    disconnect(server.id)
    setStatus(server.id, { state: "error", message: describeError(error) })
  }
}

/** Checks enabled servers this session hasn't reached yet. */
export function checkMcpServers() {
  const status = mcpStatusStore.get()
  for (const server of getMcpServers()) {
    if (server.enabled && !status[server.id]) void checkMcpServer(server)
  }
}

/**
 * Small models have small context windows, so long tool results are cut to
 * this many characters.
 */
const MAX_RESULT_CHARS = 6000

type CallToolResult = {
  content?: Array<{ type: string; text?: string }>
  structuredContent?: unknown
  isError?: boolean
}

/** Runs one MCP tool and returns its result as text for the model. */
async function runMcpTool(server: McpServer, name: string, input: unknown) {
  const client = await connect(server)
  const result = (await client.callTool(name, {
    ...(input as Record<string, unknown> | undefined),
    ...server.fixedArgs
  })) as CallToolResult
  const text =
    result.content
      ?.map((c) => (c.type === "text" ? (c.text ?? "") : `[${c.type}]`))
      .join("\n") ||
    (result.structuredContent === undefined
      ? ""
      : JSON.stringify(result.structuredContent))
  if (result.isError) throw new Error(text || "The tool failed.")
  return shorten(compactToolResult(text))
}

/** Cuts a result to MAX_RESULT_CHARS, keeping a trailing Sources list whole. */
function shorten(text: string) {
  if (text.length <= MAX_RESULT_CHARS) return text
  const at = text.lastIndexOf("\n\nSources:\n")
  const tail = at >= 0 ? text.slice(at) : ""
  const head = at >= 0 ? text.slice(0, at) : text
  const room = Math.max(0, MAX_RESULT_CHARS - tail.length)
  return `${head.slice(0, room)}\n…(cut to ${MAX_RESULT_CHARS} characters)${tail}`
}

/**
 * The tools of every enabled, connected server, as TanStack AI client tools:
 * useChat runs them in the browser when the model calls one. When two
 * servers have a tool with the same name, the first one wins.
 */
export function useMcpTools() {
  const servers = useMcpServers()
  const status = mcpStatusStore.use()
  return useMemo(() => {
    const seen = new Set<string>()
    // Built-in servers come first in the list, so their tools lead.
    return servers.flatMap((server) => {
      const s = status[server.id]
      if (!server.enabled || s?.state !== "connected") return []
      return s.tools
        .filter((tool) => !seen.has(tool.name) && seen.add(tool.name))
        .map((tool) =>
          toolDefinition({
            name: tool.name,
            description: tool.description,
            inputSchema: tool.inputSchema
          }).client((input) => runMcpTool(server, tool.name, input))
        )
    })
  }, [servers, status])
}

/** A default name for a server: its host, without a leading "mcp.". */
export function serverNameFromUrl(url: string) {
  try {
    return new URL(url).hostname.replace(/^mcp\./, "")
  } catch {
    return url
  }
}

export function addMcpServer(input: {
  name: string
  url: string
  token?: string
}) {
  const server: McpServer = {
    id: uuidv7(),
    name: input.name.trim() || serverNameFromUrl(input.url),
    url: input.url.trim(),
    token: input.token?.trim() || undefined,
    enabled: true
  }
  customServersStore.set((all) => [...all, server])
  void checkMcpServer(server)
  return server
}

export function setMcpServerEnabled(id: string, enabled: boolean) {
  if (!enabled) disconnect(id)
  if (BUILT_IN_SERVERS.some((s) => s.id === id)) {
    builtInEnabledStore.set((all) => ({ ...all, [id]: enabled }))
    return
  }
  customServersStore.set((all) =>
    all.map((s) => (s.id === id ? { ...s, enabled } : s))
  )
}

/** Removes a server the user added (built-in ones stay). */
export function removeMcpServer(id: string) {
  if (BUILT_IN_SERVERS.some((s) => s.id === id)) return
  disconnect(id)
  customServersStore.set((all) => all.filter((s) => s.id !== id))
  mcpStatusStore.set(({ [id]: _removed, ...rest }) => rest)
}
