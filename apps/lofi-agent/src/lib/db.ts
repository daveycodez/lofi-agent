/**
 * The app's data, in TanStack DB collections persisted to IndexedDB: chat
 * sessions and their messages, settings, MCP servers and reply stats. One
 * IndexedDB database holds a store per collection. Collections load whole
 * stores into memory, so reads are synchronous once `openDb()` resolves,
 * and changes reach other open tabs.
 *
 * Browser only: the route that uses this turns server rendering off and
 * awaits `openDb()` in its loader.
 */
import type { ChatPersistedState } from "@tanstack/ai-react"
import {
  collectionOptions,
  createIndexedDB,
  DbClient,
  indexedDBCollectionOptions
} from "@tanstack/react-db"
import type { IndexedDBInstance } from "@tanstack/react-db"

import type { ReplyStats } from "#/chat/generation-stats"
import type { ModelId } from "#/chat/models"

export type Thread = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  // Always set: live queries compare them, and undefined never matches.
  archived: boolean
  pinned: boolean
}

/** A session's messages, as useChat persists them, keyed by thread id. */
export type ChatRecord = { id: string; state: ChatPersistedState }

/** Every setting, by key, with its value's type. */
export type Settings = {
  /** A system prompt the user wrote, sent before the date. Empty means none. */
  systemPrompt: string
  model: ModelId
  /** Whether models with a thinking switch reason before answering. */
  think: boolean
  sidebarOpen: boolean
}

export type SettingKey = keyof Settings
export type SettingRecord = { id: SettingKey; value: unknown }

/**
 * An MCP server the user added, or a built-in server's saved choices: only
 * `enabled` and `token` (its key, as the user entered it) are kept for those.
 */
export type McpServerRecord = {
  id: string
  enabled: boolean
  name?: string
  url?: string
  /** Sent with every request: servers take keys in different headers. */
  headers?: Record<string, string>
  token?: string
}

/** A session's latest reply's stats, keyed by thread id. */
export type ReplyStatsRecord = { id: string; stats: ReplyStats }

const DATABASE = "lofi-agent-app"
const STORES = ["threads", "chats", "settings", "mcpServers", "replyStats"]

const idb = (client: DbClient) =>
  client.requireDependency<IndexedDBInstance>("idb")

export const threadsCollection = collectionOptions("threads", (client) =>
  indexedDBCollectionOptions<Thread, string>({
    id: "threads",
    db: idb(client),
    name: "threads",
    getKey: (thread) => thread.id
  })
)

export const chatsCollection = collectionOptions("chats", (client) =>
  indexedDBCollectionOptions<ChatRecord, string>({
    id: "chats",
    db: idb(client),
    name: "chats",
    getKey: (chat) => chat.id
  })
)

export const settingsCollection = collectionOptions("settings", (client) =>
  indexedDBCollectionOptions<SettingRecord, string>({
    id: "settings",
    db: idb(client),
    name: "settings",
    getKey: (setting) => setting.id
  })
)

export const mcpServersCollection = collectionOptions("mcpServers", (client) =>
  indexedDBCollectionOptions<McpServerRecord, string>({
    id: "mcpServers",
    db: idb(client),
    name: "mcpServers",
    getKey: (server) => server.id
  })
)

export const replyStatsCollection = collectionOptions("replyStats", (client) =>
  indexedDBCollectionOptions<ReplyStatsRecord, string>({
    id: "replyStats",
    db: idb(client),
    name: "replyStats",
    getKey: (record) => record.id
  })
)

let client: DbClient | undefined
let opening: Promise<DbClient> | undefined

// Editing this module in development reloads the page: a hot swap would
// leave the open client behind in the old module.
if (import.meta.hot) import.meta.hot.accept(() => window.location.reload())

/** Opens the database and loads every collection. Safe to call again. */
export function openDb() {
  opening ??= (async () => {
    const db = await createIndexedDB({ name: DATABASE, version: 1, stores: STORES })
    const next = new DbClient({ idb: db })
    await Promise.all([
      next.collection(threadsCollection).preload(),
      next.collection(chatsCollection).preload(),
      next.collection(settingsCollection).preload(),
      next.collection(mcpServersCollection).preload(),
      next.collection(replyStatsCollection).preload()
    ])
    client = next
    return next
  })()
  // A failed open isn't kept, so the next navigation tries again.
  opening.catch(() => {
    opening = undefined
  })
  return opening
}

/** The browser's DbClient. Throws before `openDb()` has resolved. */
export function dbClient() {
  if (!client) throw new Error("The database isn't open yet: await openDb().")
  return client
}

export const threads = () => dbClient().collection(threadsCollection)
export const chats = () => dbClient().collection(chatsCollection)
export const settings = () => dbClient().collection(settingsCollection)
export const mcpServers = () => dbClient().collection(mcpServersCollection)
export const replyStats = () => dbClient().collection(replyStatsCollection)

/** Deletes a row if it's there. */
export function deleteIfPresent(
  collection: { has: (key: string) => boolean; delete: (key: string) => unknown },
  key: string
) {
  if (collection.has(key)) collection.delete(key)
}
