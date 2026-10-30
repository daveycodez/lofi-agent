/**
 * Chat sessions: their metadata in the threads collection, and their
 * messages in the chats collection (written by useChat through
 * `chatPersistence`, in src/chat/persistence.ts).
 */
import { forgetReplyStats } from "#/chat/generation-stats"
import { forgetChat } from "#/chat/persistence"
import { deleteIfPresent, threads } from "#/lib/db"

export function titleFrom(text: string) {
  const line = text.trim().split("\n")[0] ?? ""
  return line.length > 48 ? `${line.slice(0, 47)}…` : line || "New session"
}

/** Records activity on a thread, creating it on first use. */
export function touchThread(id: string, firstMessage?: string) {
  const collection = threads()
  const now = Date.now()
  if (collection.has(id)) {
    collection.update(id, (draft) => {
      draft.updatedAt = now
      draft.archived = false
    })
    return
  }
  collection.insert({
    id,
    title: titleFrom(firstMessage ?? ""),
    createdAt: now,
    updatedAt: now,
    archived: false,
    pinned: false
  })
}

export function renameThread(id: string, title: string) {
  const clean = title.trim()
  if (!clean || !threads().has(id)) return
  threads().update(id, (draft) => {
    draft.title = clean
  })
}

export function togglePinThread(id: string) {
  if (!threads().has(id)) return
  threads().update(id, (draft) => {
    draft.pinned = !draft.pinned
  })
}

export function archiveThread(id: string) {
  if (!threads().has(id)) return
  threads().update(id, (draft) => {
    draft.archived = true
  })
}

/** Deletes a session: its metadata, messages and stats. */
export function deleteThread(id: string) {
  deleteIfPresent(threads(), id)
  forgetChat(id)
  forgetReplyStats([id])
}

/** Deletes every session, archived ones included, with their messages. */
export function deleteAllThreads() {
  const ids = threads().toArray.map((t) => t.id)
  for (const id of ids) deleteThread(id)
}
