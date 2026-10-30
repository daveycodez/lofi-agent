/** User settings, kept in this browser's storage. */
import { createStore } from "#/lib/store"

/** Whether models get the built-in web search tool. On by default. */
export const webSearchEnabledStore = createStore(true, "lofi-agent:web-search")

/**
 * A system prompt the user wrote, sent to every model at the start of each
 * conversation, after the current date. Empty means none.
 */
export const systemPromptStore = createStore("", "lofi-agent:system-prompt")

/**
 * The system prompt sent to every model: today's date, so models know when
 * "now" is, then the user's own prompt if they wrote one. The date only, not
 * the time: the prompt then stays the same all day, so the local model can
 * reuse what it already read of a chat instead of reading it all again.
 */
export function currentSystemPrompt() {
  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric"
  })
  return [`Today's date is ${today}.`, systemPromptStore.get().trim()]
    .filter(Boolean)
    .join("\n\n")
}
