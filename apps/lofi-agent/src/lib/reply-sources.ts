/**
 * The web sources a reply's tools returned, for the citation chips in its
 * answer and the source list under it.
 */
import type { UIMessage } from "@tanstack/ai"

import { parseSources } from "#/lib/sources"
import type { Source } from "#/lib/sources"

export type ReplySources = {
  /** Every source the reply's tools returned, once each. */
  all: Array<Source>
  /** The latest tool result's sources, which `[n]` in the answer refers to. */
  cited: Array<Source>
}

/** The sources in each tool result of a message, in order. */
function sourceListsOf(message: UIMessage) {
  return message.parts
    .map((p) =>
      p.type === "tool-call" && typeof p.output === "string"
        ? parseSources(p.output)
        : p.type === "tool-result" && typeof p.content === "string"
          ? parseSources(p.content)
          : []
    )
    .filter((list) => list.length > 0)
}

/**
 * Sources for each reply, keyed by the reply's last message. A reply that
 * called tools spans several assistant messages: the calls, then the answer.
 */
export function sourcesByReply(messages: ReadonlyArray<UIMessage>) {
  const byReply = new Map<string, ReplySources>()
  let lists: Array<Array<Source>> = []
  messages.forEach((message, i) => {
    if (message.role !== "assistant") {
      lists = []
      return
    }
    lists.push(...sourceListsOf(message))
    if (messages[i + 1]?.role === "assistant" || lists.length === 0) return
    const seen = new Set<string>()
    byReply.set(message.id, {
      all: lists.flat().filter((s) => !seen.has(s.url) && seen.add(s.url)),
      cited: lists.at(-1) ?? []
    })
  })
  return byReply
}

/** A message's text parts, joined. */
export function messageText(message: UIMessage) {
  return message.parts.map((p) => (p.type === "text" ? p.content : "")).join("")
}
