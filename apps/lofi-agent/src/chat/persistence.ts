/**
 * useChat's persistence: each thread's messages are one row in the chats
 * TanStack DB collection, which persists to IndexedDB.
 */
import type { ChatPersistedState, ChatStorageAdapter } from "@tanstack/ai-react"

import { chats, deleteIfPresent, openDb } from "#/lib/db"

/**
 * Sessions deleted since the page loaded. The open session's chat client
 * writes once more as it unmounts, and a running reply keeps writing until
 * it stops; those late writes must not bring a deleted session's messages
 * back. Thread ids are never reused.
 */
const deletedThreads = new Set<string>()

/**
 * One row per thread in the chats collection. useChat writes the whole
 * transcript on each change, in order, waiting for each write. Rows are
 * copied in and out, so the chat client and the collection never share
 * objects (collection rows must not be changed in place).
 *
 * Each method waits for the database rather than throwing: useChat treats
 * a failed read as an empty chat, and its next write of that empty chat
 * would remove the stored one.
 */
export const chatPersistence: ChatStorageAdapter<ChatPersistedState> = {
  async getItem(id) {
    await openDb()
    const row = chats().get(id)
    return row ? structuredClone(row.state) : null
  },
  async setItem(id, value) {
    if (deletedThreads.has(id)) return
    const state = structuredClone(value)
    await openDb()
    const collection = chats()
    const tx = collection.has(id)
      ? collection.update(id, (draft) => {
          draft.state = state
        })
      : collection.insert({ id, state })
    await tx.when("settled")
  },
  async removeItem(id) {
    await openDb()
    const collection = chats()
    if (collection.has(id)) await collection.delete(id).when("settled")
  }
}

/** Deletes a thread's messages for good, late writes included. */
export function forgetChat(id: string) {
  deletedThreads.add(id)
  deleteIfPresent(chats(), id)
}

/** A thread's messages, for copying it as text. */
export const threadMessages = (id: string) =>
  chats().get(id)?.state.messages ?? []

/** Whether a thread was saved waiting for an approval. */
export const hasPendingInterrupts = (id: string) =>
  (chats().get(id)?.state.resume?.pendingInterrupts?.length ?? 0) > 0
