/** User settings, kept in this browser's storage. */
import { createStore } from "#/lib/store"

/** Whether models get the built-in web search tool. On by default. */
export const webSearchEnabledStore = createStore(true, "lofi-agent:web-search")

/**
 * A system prompt the user wrote, sent to every model at the start of each
 * conversation, after the current date and time. Empty means none.
 */
export const systemPromptStore = createStore("", "lofi-agent:system-prompt")

/**
 * The system prompt sent to every model: the current date and time, so
 * models know when "now" is, then the user's own prompt if they wrote one.
 */
export function currentSystemPrompt() {
  const now = new Date().toLocaleString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short"
  })
  return [
    `The current date and time is ${now}.`,
    systemPromptStore.get().trim()
  ]
    .filter(Boolean)
    .join("\n\n")
}
