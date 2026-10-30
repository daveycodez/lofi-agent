/**
 * MCP servers: the built-in ones (Exa's web search), then those the user
 * adds. Each enabled server gets one TanStack AI MCP client
 * (`createMCPClient`), kept open, and chat() discovers and runs their tools
 * (`chat({ mcp: { clients, connection: "keep-alive" } })`, see
 * src/chat/connection.ts).
 *
 * TanStack's docs keep MCP clients on a server. This app has none: models
 * run in the browser, so the browser connects to each server directly
 * (Streamable HTTP), and a server only works if it allows requests from
 * this site (CORS). The list, on/off choices and any keys stay in this
 * browser, in the mcpServers collection.
 *
 * Approval follows TanStack's rule for server trust: a tool runs without
 * asking only when it comes from a built-in server (which this app trusts)
 * and the server marks it read-only. Every tool of a server the user added
 * asks first, whatever its hints say: hints are only labels.
 */
import { toolDefinition } from "@tanstack/ai"
import type { MCPClient, MCPClientOptions, McpServerTool } from "@tanstack/ai-mcp"
import { useLiveQuery, useOptionalDbClient } from "@tanstack/react-db"
import { useMemo } from "react"
import { v7 as uuidv7 } from "uuid"

import { deleteIfPresent, mcpServers, mcpServersCollection } from "#/lib/db"
import type { McpServerRecord } from "#/lib/db"
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
  /**
   * A built-in server known to speak only the older protocol: connect with
   * its handshake straight away, rather than after a blocked first try.
   */
  legacyProtocol?: boolean
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
  /** The name the model sees: a server the user added prefixes it. */
  name: string
  /** The server's display name for it. */
  title: string
  description: string
  /** The run pauses for the user's approval before each call. */
  needsApproval: boolean
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
    tokenLabel: "Exa API key",
    legacyProtocol: true
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

/** The servers, built-in ones only until the database is open. */
export function useMcpServers() {
  const db = useOptionalDbClient()
  const { data } = useLiveQuery({
    query: (q) => (db ? q.from({ server: mcpServersCollection }) : null)
  })
  return useMemo(() => allServers(data ?? []), [data])
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

/** Every message down an error's cause chain. */
function errorMessages(error: unknown) {
  // The client wraps failures ("Failed to connect to MCP server"); the
  // reason is further down the cause chain.
  const messages: Array<string> = []
  for (let e = error, depth = 0; e != null && depth < 5; depth++) {
    messages.push(e instanceof Error ? e.message : String(e))
    e = e instanceof Error ? e.cause : undefined
  }
  return messages
}

/**
 * Chrome says "Failed to fetch" and Safari "Load failed" when a request is
 * blocked, which for a reachable server almost always means CORS.
 */
const wasBlocked = (error: unknown) =>
  /failed to fetch|load failed|networkerror/i.test(
    errorMessages(error).join(" | ")
  )

/** A readable reason a connection failed. */
function describeError(error: unknown) {
  const messages = errorMessages(error)
  const message = messages.join(" | ")
  if (wasBlocked(error)) {
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

/**
 * The prefix for a server the user added, from its name, so its tools can't
 * collide with another server's (chat() fails a run on a duplicate name).
 * A later server with the same name gets a number. Built-in servers keep
 * their tools' own names.
 */
function prefixOf(server: McpServer) {
  if (server.builtIn) return undefined
  const slug = (name: string) =>
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 20) || "mcp"
  const own = slug(server.name)
  const same = getMcpServers().filter(
    (s) => !s.builtIn && slug(s.name) === own
  )
  const index = same.findIndex((s) => s.id === server.id)
  return index > 0 ? `${own}_${index + 1}` : own
}

/** How to connect to a server, and which of its tools ask first. */
function clientOptions(server: McpServer): MCPClientOptions {
  const prefix = prefixOf(server)
  return {
    transport: { type: "http", url: server.url, headers: headersOf(server) },
    name: "lofi-agent",
    ...(prefix && { prefix }),
    ...(server.legacyProtocol && { prior: { kind: "legacy" as const } }),
    needsApproval: (tool) =>
      !(server.builtIn === true && tool.annotations?.readOnlyHint === true)
  }
}

/** Open clients by server id, kept for every run of the session. */
const clients = new Map<string, Promise<MCPClient>>()
/** The connected ones, and which server each belongs to. */
const connected = new Map<string, MCPClient>()
/** Each connected server's tools, as its client lists them. */
const serverTools = new Map<string, Array<McpServerTool>>()
const serverOfClient = new WeakMap<object, string>()

async function createClient(server: McpServer) {
  const { createMCPClient } = await import("@tanstack/ai-mcp")
  const options = clientOptions(server)
  try {
    return await createMCPClient(options)
  } catch (error) {
    if (!wasBlocked(error)) throw error
    // The client first asks for the newest protocol, with headers that
    // servers whose CORS rules predate them don't allow (Exa's), so the
    // browser blocks it. The older handshake sends none of them.
    return await createMCPClient({ ...options, prior: { kind: "legacy" } })
  }
}

function connect(server: McpServer) {
  let client = clients.get(server.id)
  if (!client) {
    client = createClient(server)
    clients.set(server.id, client)
    // A failed connection isn't kept, so the next try reconnects.
    client.then(
      (c) => serverOfClient.set(c, server.id),
      () => clients.delete(server.id)
    )
  }
  return client
}

function disconnect(id: string) {
  const client = clients.get(id)
  clients.delete(id)
  connected.delete(id)
  serverTools.delete(id)
  void client?.then((c) => c.close()).catch(() => undefined)
}

/** Connects to `server`, lists its tools, and records the result. */
export async function checkMcpServer(server: McpServer) {
  disconnect(server.id)
  setStatus(server.id, { state: "checking" })
  try {
    const client = await connect(server)
    const tools = await client.tools()
    connected.set(server.id, client)
    serverTools.set(server.id, tools)
    setStatus(server.id, {
      state: "connected",
      tools: tools.map((t) => ({
        name: t.name,
        title: t.metadata.mcp.title,
        description: t.description,
        // Set at runtime from the client's `needsApproval`; the static type
        // only knows tools declared ahead of time.
        needsApproval: Boolean(t.needsApproval)
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
 * The clients of every enabled, connected server, for `chat({ mcp })`.
 * Built-in servers come first, so their tools lead (small models tend to
 * pick the first tool that might fit).
 */
export const getMcpClients = () =>
  getMcpServers().flatMap((server) => {
    const client = connected.get(server.id)
    return server.enabled && client ? [client] : []
  })

/**
 * Definitions of the tools that ask first, for useChat's `tools`. The chat
 * client only accepts an approval for a tool it has the definition of, with
 * the same input schema (TanStack's approval guide: "pass the definition to
 * useChat"). They come from the same tools chat() runs, so the schemas
 * match; they carry no `execute`, so nothing runs them in the client.
 */
export function useMcpApprovalTools() {
  const servers = useMcpServers()
  const status = mcpStatusStore.use()
  return useMemo(
    () =>
      servers.flatMap((server) =>
        server.enabled && status[server.id]?.state === "connected"
          ? (serverTools.get(server.id) ?? [])
              .filter((tool) => tool.needsApproval)
              .map((tool) =>
                toolDefinition({
                  name: tool.name,
                  description: tool.description,
                  inputSchema: tool.inputSchema,
                  needsApproval: true
                })
              )
          : []
      ),
    [servers, status]
  )
}

/** Whether every enabled server has connected or failed, so its tools are known. */
export function useMcpSettled() {
  const servers = useMcpServers()
  const status = mcpStatusStore.use()
  return servers.every((server) => {
    const state = status[server.id]?.state
    return !server.enabled || state === "connected" || state === "error"
  })
}

/**
 * A client whose tools couldn't be listed for a run: the run goes on
 * without them, and the server shows the error in the MCP servers list.
 */
export function reportMcpDiscoveryError(error: unknown, source: object) {
  const id = serverOfClient.get(source)
  if (!id) return
  connected.delete(id)
  setStatus(id, { state: "error", message: describeError(error) })
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
