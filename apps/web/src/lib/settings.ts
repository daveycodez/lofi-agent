/** User settings, kept in this browser's storage. */
import { createStore } from "#/lib/store"

/** Whether models get the built-in web search tool. On by default. */
export const webSearchEnabledStore = createStore(
  true,
  "lofi-agent:web-search"
)

/**
 * A system prompt the user wrote, sent to every model at the start of each
 * conversation. Empty means none: each model runs as its model card shows.
 */
export const systemPromptStore = createStore("", "lofi-agent:system-prompt")

/** The system prompt to send, or undefined when there is none. */
export function currentSystemPrompt() {
  return systemPromptStore.get().trim() || undefined
}
