/** User settings, one row each in the settings collection. */
import { eq, useLiveQuery } from "@tanstack/react-db"

import { settings, settingsCollection } from "#/lib/db"
import type { SettingKey, Settings } from "#/lib/db"

/**
 * One setting: `get` for code outside React, `use` to re-render when it
 * changes, `set` to save it. A setting never saved reads as `initial`.
 */
export function setting<K extends SettingKey>(key: K, initial: Settings[K]) {
  const get = (): Settings[K] => {
    const row = settings().get(key)
    return row ? (row.value as Settings[K]) : initial
  }

  const set = (value: Settings[K]) => {
    const collection = settings()
    if (collection.has(key)) {
      collection.update(key, (draft) => {
        draft.value = value
      })
    } else {
      collection.insert({ id: key, value })
    }
  }

  const use = (): Settings[K] => {
    const { data } = useLiveQuery({
      query: (q) =>
        q
          .from({ setting: settingsCollection })
          .where(({ setting }) => eq(setting.id, key))
          .findOne()
    })
    return data ? (data.value as Settings[K]) : initial
  }

  return { get, set, use }
}

/**
 * A system prompt the user wrote, sent to every model at the start of each
 * conversation, before the current date. Empty means none.
 */
export const systemPromptSetting = setting("systemPrompt", "")

/**
 * The system prompt sent to every model: the user's own prompt if they wrote
 * one, then today's date, so models know when "now" is. The date comes last
 * and is the date only, not the time, so a new minute doesn't change the
 * prompt, and the local model reuses what it already read of it.
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
  return [systemPromptSetting.get().trim(), `The current date is ${today}.`]
    .filter(Boolean)
    .join("\n\n")
}
