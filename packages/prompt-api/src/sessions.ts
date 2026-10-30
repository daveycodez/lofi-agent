/**
 * Sessions by conversation. A Prompt API session holds the conversation
 * it has read (the model's cache), so each thread keeps its own: when the
 * next request extends what the session has read (a follow-up, or the step
 * after a tool call), only the new messages are read. Anything else (an
 * edited or regenerated message, another system prompt) starts a new
 * session from the whole conversation.
 */
import { languageModel } from "./types.js"
import type {
  LanguageModelCreateOptions,
  LanguageModelMessage,
  LanguageModelSession
} from "./types.js"
import { noteContextWindow } from "./model.js"

/** Sessions kept at once, most recently used first; older ones are freed. */
const MAX_SESSIONS = 4

type Entry = {
  session: LanguageModelSession
  /** The system prompt and create options the session started with. */
  setup: string
  /** The messages the session has read, after the system prompt. */
  history: Array<string>
  busy: boolean
}

const entries = new Map<string, Entry>()

const keyOf = (m: LanguageModelMessage) => `${m.role}\u0000${m.content}`

const startsWith = (list: Array<string>, start: Array<string>) =>
  start.length <= list.length && start.every((key, i) => list[i] === key)

/** A session lent to one model call. */
export type Lease = {
  session: LanguageModelSession
  /** Tokens the session had already read and didn't read again. */
  cachedTokens: number
  /** Time spent creating the session (reading the conversation), in ms. */
  createMs: number
  /** Gives the session back, having read `history` (after the system prompt). */
  release: (history: Array<LanguageModelMessage>) => void
  /** Frees the session: after an abort or error, what it holds is unknown. */
  discard: () => void
}

/**
 * A session for a conversation that has read `system` and `prior`: the
 * thread's own if it has read a start of them (the rest is appended), else
 * a new one.
 */
export async function leaseSession({
  threadId,
  system,
  prior,
  options,
  signal
}: {
  threadId: string | undefined
  system: string
  prior: Array<LanguageModelMessage>
  options: LanguageModelCreateOptions
  signal: AbortSignal | undefined
}): Promise<Lease> {
  const setup = JSON.stringify({ system, options })
  const keys = prior.map(keyOf)
  const entry = threadId ? entries.get(threadId) : undefined

  if (
    entry &&
    !entry.busy &&
    entry.setup === setup &&
    startsWith(keys, entry.history)
  ) {
    entry.busy = true
    // Most recently used last.
    entries.delete(threadId as string)
    entries.set(threadId as string, entry)
    const cachedTokens = entry.session.contextUsage
    const started = performance.now()
    const gap = prior.slice(entry.history.length)
    try {
      if (gap.length > 0) await entry.session.append(gap, { signal })
    } catch (error) {
      drop(threadId as string, entry)
      throw error
    }
    entry.history = keys
    return lease(threadId, entry, cachedTokens, performance.now() - started)
  }

  // Another run of this thread holds its session (it shouldn't happen):
  // this one gets a session of its own, freed when it ends.
  const shared = !entry?.busy
  if (entry && shared && threadId) drop(threadId, entry)
  const started = performance.now()
  const session = await createSession(system, prior, options, signal)
  const created: Entry = { session, setup, history: keys, busy: true }
  if (threadId && shared) {
    entries.set(threadId, created)
    for (const [id, old] of entries) {
      if (entries.size <= MAX_SESSIONS) break
      if (!old.busy) drop(id, old)
    }
  }
  return lease(
    threadId && shared ? threadId : undefined,
    created,
    0,
    performance.now() - started
  )
}

function lease(
  threadId: string | undefined,
  entry: Entry,
  cachedTokens: number,
  createMs: number
): Lease {
  return {
    session: entry.session,
    cachedTokens,
    createMs,
    release: (history) => {
      if (!threadId) return entry.session.destroy()
      entry.history = history.map(keyOf)
      entry.busy = false
    },
    discard: () => {
      if (threadId) drop(threadId, entry)
      else entry.session.destroy()
    }
  }
}

function drop(threadId: string, entry: Entry) {
  if (entries.get(threadId) === entry) entries.delete(threadId)
  entry.session.destroy()
}

const isQuotaError = (error: unknown) =>
  error instanceof DOMException && error.name === "QuotaExceededError"

/**
 * A new session that has read the system prompt and the conversation. If
 * the conversation doesn't fit, its oldest turns are left out until it
 * does, as the session itself does when a conversation outgrows it.
 */
async function createSession(
  system: string,
  prior: Array<LanguageModelMessage>,
  options: LanguageModelCreateOptions,
  signal: AbortSignal | undefined
) {
  const api = languageModel()
  if (!api) throw new Error("No Prompt API in this browser.")
  let rest = prior
  for (;;) {
    try {
      const session = await api.create({
        ...options,
        initialPrompts: [
          ...(system ? [{ role: "system" as const, content: system }] : []),
          ...rest
        ],
        signal
      })
      noteContextWindow(session)
      return session
    } catch (error) {
      if (!isQuotaError(error) || rest.length === 0) throw error
      // Leave out the oldest turn: up to the next user message after it.
      const next = rest.findIndex((m, i) => i > 0 && m.role === "user")
      rest = next > 0 ? rest.slice(next) : []
    }
  }
}

/** Frees every session kept, as when the page no longer needs the model. */
export function destroyPromptApiSessions() {
  for (const [id, entry] of entries) {
    if (!entry.busy) drop(id, entry)
  }
}
