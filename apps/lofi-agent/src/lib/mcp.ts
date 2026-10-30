/**
 * MCP servers the user adds. The browser connects to each one directly with
 * TanStack AI's MCP client (Streamable HTTP), so a server only works if it
 * allows requests from this site (CORS). The list, on/off choices and any
 * tokens stay in this browser's storage.
 */
import { toolDefinition } from "@tanstack/ai"
import type { JSONSchema } from "@tanstack/ai"
import type { MCPClient } from "@tanstack/ai-mcp"
import { useMemo } from "react"
import { v7 as uuidv7 } from "uuid"

import { compactToolResult, shorten } from "#/lib/sources"
import { createStore } from "#/lib/store"

export type McpServer = {
  id: string
  name: string
  url: string
  /** Sent as `Authorization: Bearer <token>` when set. */
  token?: string
  enabled: boolean
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

export const getMcpServers = () => customServersStore.get()

export const useMcpServers = () => customServersStore.use()

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
 * The tools of every enabled, connected server, as TanStack AI client tools:
 * useChat runs them in the browser when the model calls one. When two
 * servers have a tool with the same name, the first one wins.
 */
export function useMcpTools() {
  const servers = useMcpServers()
  const status = mcpStatusStore.use()
  return useMemo(() => {
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
  customServersStore.set((all) =>
    all.map((s) => (s.id === id ? { ...s, enabled } : s))
  )
}

export function removeMcpServer(id: string) {
  disconnect(id)
  customServersStore.set((all) => all.filter((s) => s.id !== id))
  mcpStatusStore.set(({ [id]: _removed, ...rest }) => rest)
}
