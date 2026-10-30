/**
 * MCP servers: the built-in ones (Exa's web search), then those the user
 * adds. The browser connects to each one directly with TanStack AI's MCP
 * client (Streamable HTTP), so a server only works if it allows requests
 * from this site (CORS). The list, on/off choices and any keys or tokens
 * stay in this browser, in the mcpServers collection.
 */
import { toolDefinition } from "@tanstack/ai"
import type { JSONSchema } from "@tanstack/ai"
import type { MCPClient } from "@tanstack/ai-mcp"
import { useLiveQuery } from "@tanstack/react-db"
import { useMemo } from "react"
import { v7 as uuidv7 } from "uuid"

import { deleteIfPresent, mcpServers, mcpServersCollection } from "#/lib/db"
import type { McpServerRecord } from "#/lib/db"
import { compactToolResult, shorten } from "#/lib/sources"
import { createStore } from "#/lib/store"

export type McpServer = {
  id: string
  name: string
  url: string
  /**
   * HTTP headers sent with every request: servers take keys in different
   * ones (`Authorization: Bearer …`, `x-api-key`, their own).
   */
  headers?: Record<string, string>
  /**
   * Older saves kept one token, sent as `Authorization: Bearer <token>`;
   * read as that header. A built-in server's key, as the user entered it.
   */
  token?: string
  enabled: boolean
  /** Comes with the app: it can be turned off, not removed. */
  builtIn?: boolean
  /** The header a built-in server takes its key in. */
  tokenHeader?: string
  /** What its key is called, for the field to enter it. */
  tokenLabel?: string
}

/** The headers to send to a server: its own, or its key in its header. */
function headersOf(server: McpServer): Record<string, string> {
  if (server.builtIn) {
    return server.token && server.tokenHeader
      ? { [server.tokenHeader]: server.token }
      : {}
  }
  if (server.headers) return server.headers
  return server.token ? { Authorization: `Bearer ${server.token}` } : {}
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

/**
 * Servers that come with the app. Exa's hosted MCP server searches the web
 * and reads pages; it works without a key at a free rate limit, and with an
 * Exa API key (sent as `x-api-key`) past it.
 */
const BUILT_IN_SERVERS: Array<Omit<McpServer, "enabled" | "token">> = [
  {
    id: "built-in:exa",
    name: "Exa",
    url: "https://mcp.exa.ai/mcp",
    builtIn: true,
    tokenHeader: "x-api-key",
    tokenLabel: "Exa API key"
  }
]

/** Built-in servers first, then the user's (minus copies of a built-in). */
function allServers(records: ReadonlyArray<McpServerRecord>): Array<McpServer> {
  const saved = new Map(records.map((r) => [r.id, r]))
  const builtIns = BUILT_IN_SERVERS.map((s) => {
    const record = saved.get(s.id)
    return {
      ...s,
      enabled: record?.enabled ?? true,
      ...(record?.token && { token: record.token })
    }
  })
  const urls = new Set(builtIns.map((s) => sameUrl(s.url)))
  const custom = records.flatMap((r): Array<McpServer> =>
    r.url && !isBuiltIn(r.id) && !urls.has(sameUrl(r.url))
      ? [
          {
            id: r.id,
            name: r.name ?? serverNameFromUrl(r.url),
            url: r.url,
            enabled: r.enabled,
            ...(r.headers && { headers: r.headers }),
            ...(r.token && { token: r.token })
          }
        ]
      : []
  )
  return [...builtIns, ...custom]
}

/** A URL in one form, to spot the same server added twice. */
function sameUrl(url: string) {
  try {
    const u = new URL(url.trim())
    return `${u.origin}${u.pathname.replace(/\/+$/, "")}${u.search}`
  } catch {
    return url.trim()
  }
}

export const getMcpServers = () => allServers(mcpServers().toArray)

export function useMcpServers() {
  const { data } = useLiveQuery({
    query: (q) => q.from({ server: mcpServersCollection })
  })
  return useMemo(() => allServers(data), [data])
}

function isBuiltIn(id: string) {
  return BUILT_IN_SERVERS.some((s) => s.id === id)
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
          headers: headersOf(server)
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

/** Connects to `server`, lists its tools, and records the result. */
export async function checkMcpServer(server: McpServer) {
  disconnect(server.id)
  setStatus(server.id, { state: "checking" })
  try {
    const tools = await (await connect(server)).tools()
    setStatus(server.id, {
      state: "connected",
      tools: tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: (t.inputSchema ?? { type: "object" }) as JSONSchema
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

type CallToolResult = {
  content?: Array<{ type: string; text?: string }>
  structuredContent?: unknown
  isError?: boolean
}

/** Runs one MCP tool and returns its result as text for the model. */
async function runMcpTool(server: McpServer, name: string, input: unknown) {
  const client = await connect(server)
  const result = (await client.callTool(
    name,
    (input ?? {}) as Record<string, unknown>
  )) as CallToolResult
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

/**
 * The tools of every enabled, connected server, as TanStack AI tools that
 * chat() runs, here in the browser. Built-in servers come first, so their
 * tools lead (small models tend to pick the first tool that might fit).
 * When two servers have a tool with the same name, the first one wins.
 */
export const getMcpTools = () => toolsOf(getMcpServers(), mcpStatusStore.get())

function toolsOf(
  servers: Array<McpServer>,
  status: Record<string, McpServerStatus>
) {
  const seen = new Set<string>()
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
          // chat() runs it, here in the page.
        }).server((input) => runMcpTool(server, tool.name, input))
      )
  })
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
  headers?: Array<{ name: string; value: string }>
}) {
  const headers = Object.fromEntries(
    (input.headers ?? [])
      .map((h) => [h.name.trim(), h.value.trim()] as const)
      .filter(([name]) => name)
  )
  const server: McpServer = {
    id: uuidv7(),
    name: input.name.trim() || serverNameFromUrl(input.url),
    url: input.url.trim(),
    ...(Object.keys(headers).length > 0 && { headers }),
    enabled: true
  }
  mcpServers().insert({
    id: server.id,
    name: server.name,
    url: server.url,
    enabled: true,
    ...(server.headers && { headers: server.headers })
  })
  void checkMcpServer(server)
  return server
}

export function setMcpServerEnabled(id: string, enabled: boolean) {
  if (!enabled) disconnect(id)
  const collection = mcpServers()
  if (collection.has(id)) {
    collection.update(id, (draft) => {
      draft.enabled = enabled
    })
  } else if (isBuiltIn(id)) {
    collection.insert({ id, enabled })
  }
}

/**
 * Saves (or clears) the key for a built-in server and reconnects with it,
 * so its tools are listed with the key's access.
 */
export function setBuiltInMcpToken(id: string, token: string) {
  const clean = token.trim()
  const collection = mcpServers()
  if (collection.has(id)) {
    collection.update(id, (draft) => {
      if (clean) draft.token = clean
      else delete draft.token
    })
  } else if (clean) {
    collection.insert({ id, enabled: true, token: clean })
  }
  const server = getMcpServers().find((s) => s.id === id)
  if (server?.enabled) void checkMcpServer(server)
}

/** Removes a server the user added (built-in ones stay). */
export function removeMcpServer(id: string) {
  if (isBuiltIn(id)) return
  disconnect(id)
  deleteIfPresent(mcpServers(), id)
  mcpStatusStore.set(({ [id]: _removed, ...rest }) => rest)
}
