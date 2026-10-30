import { browserChatConnection } from "#/chat/connection"
import { chatPersistence } from "#/chat/persistence"

/**
 * useChat's options, shared by every session. Replies run in the browser
 * (see ./connection.ts), so there is no server route; messages persist to
 * IndexedDB through a TanStack DB collection (see ./persistence.ts).
 */
export const chatOptions = {
  connection: browserChatConnection,
  persistence: chatPersistence
}
