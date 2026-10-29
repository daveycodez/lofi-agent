import { indexedDBPersistence } from "@tanstack/ai-react"

import { createStore } from "#/lib/store"

/** Messages for each thread live in IndexedDB, keyed by thread id. */
export const chatPersistence = indexedDBPersistence({
  databaseName: "lofi-agent"
})

export type ThreadMeta = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  archived?: boolean
  pinned?: boolean
}

/** The thread list itself is small metadata, kept in localStorage for a synchronous sidebar. */
export const threadsStore = createStore<Array<ThreadMeta>>(
  [],
  "lofi-agent:threads"
)

export function titleFrom(text: string) {
  const line = text.trim().split("\n")[0] ?? ""
  return line.length > 48 ? `${line.slice(0, 47)}…` : line || "New chat"
}

/** Records activity on a thread, creating it on first use. */
export function touchThread(id: string, firstMessage?: string) {
  threadsStore.set((threads) => {
    const now = Date.now()
    const existing = threads.find((t) => t.id === id)
    if (existing) {
      return threads.map((t) =>
        t.id === id ? { ...t, updatedAt: now, archived: false } : t
      )
    }
    return [
      {
        id,
        title: titleFrom(firstMessage ?? ""),
        createdAt: now,
        updatedAt: now
      },
      ...threads
    ]
  })
}

export function renameThread(id: string, title: string) {
  const clean = title.trim()
  if (!clean) return
  threadsStore.set((threads) =>
    threads.map((t) => (t.id === id ? { ...t, title: clean } : t))
  )
}

export function togglePinThread(id: string) {
  threadsStore.set((threads) =>
    threads.map((t) => (t.id === id ? { ...t, pinned: !t.pinned } : t))
  )
}

export function archiveThread(id: string) {
  threadsStore.set((threads) =>
    threads.map((t) => (t.id === id ? { ...t, archived: true } : t))
  )
}

export async function deleteThread(id: string) {
  threadsStore.set((threads) => threads.filter((t) => t.id !== id))
  try {
    await chatPersistence.removeItem(id)
  } catch {
    // Storage unavailable: the metadata is gone, which is what the user sees.
  }
}
