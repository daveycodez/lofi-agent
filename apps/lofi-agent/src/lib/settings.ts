/** User settings, kept in this browser's storage. */
import { createStore } from "#/lib/store"

/**
 * A system prompt the user wrote, sent to every model at the start of each
 * conversation, before the current date. Empty means none.
 */
export const systemPromptStore = createStore("", "lofi-agent:system-prompt")

/**
 * The system prompt sent to every model: the user's own prompt if they wrote
 * one, then today's date, so models know when "now" is. The date comes last
 * (the local model's worker also moves it after the tool definitions) and
 * is the date only, not the time, so a new day or minute doesn't change the
 * start of the prompt, which the local model reuses instead of reading it
 * all again.
 * The wording matters: with "Today's date is …", LFM3 thinking about a
 * question on recent events decided it had no current information and
 * didn't call web search; "The current date is …" doesn't do that.
 */
export function currentSystemPrompt() {
  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric"
  })
  return [systemPromptStore.get().trim(), `The current date is ${today}.`]
    .filter(Boolean)
    .join("\n\n")
}
