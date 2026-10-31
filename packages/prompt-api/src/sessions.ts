/**
 * Sessions by conversation. A Prompt API session holds the conversation
 * it has read (the model's cache), so each thread keeps its own: when the
 * next request extends what the session has read (a follow-up, or the step
 * after a tool call), only the new messages are read. Anything else (an
 * edited or regenerated message, another system prompt) starts a new
 * session from the whole conversation.
 */
import { createLanguageModel } from "./model.js"
import { keyOf, toLanguageModelMessages } from "./messages.js"
import type { PromptMessage } from "./messages.js"
import type {
  LanguageModelCreateOptions,
  LanguageModelSession
} from "./types.js"

/** Sessions kept at once, most recently used first; older ones are freed. */
const MAX_SESSIONS = 4

type Entry = {
  session: LanguageModelSession
  /** The system prompt and create options the session started with. */
  setup: string
  /** The messages the session has read, after the system prompt. */
  history: Array<string>
  busy: boolean
  /** Set when the browser drops old turns to make room; read per call. */
  overflowed: boolean
}

const entries = new Map<string, Entry>()

const startsWith = (list: Array<string>, start: Array<string>) =>
  start.length <= list.length && start.every((key, i) => list[i] === key)

/** A session lent to one model call. */
export type Lease = {
  session: LanguageModelSession
  /** Tokens the session had already read and didn't read again. */
  cachedTokens: number
  /** Time spent creating the session (reading the conversation), in ms. */
  createMs: number
  /** Why the thread's kept session couldn't be used, if it had one. */
  miss?: string
  /** True for the thread's kept session, false for one made for this call. */
  reused: boolean
  /** Whether the browser dropped old turns to make room since the lease began. */
  overflowed: () => boolean
  /** Gives the session back, having read `history` (after the system prompt). */
  release: (history: Array<PromptMessage>) => void
  /** Frees the session: after an abort or error, what it holds is unknown. */
  discard: () => void
}

/**
 * A session for a conversation that has read `system` and `prior`: the
 * thread's own if it has read a start of them (the rest is appended), else
 * a new one. `media` caches the images and audio loaded for the call.
 */
export async function leaseSession({
  threadId,
  system,
  prior,
  options,
  signal,
  media
}: {
  threadId: string | undefined
  system: string
  prior: Array<PromptMessage>
  options: LanguageModelCreateOptions
  signal: AbortSignal | undefined
  media?: Map<string, Promise<Blob>>
}): Promise<Lease> {
  const setup = JSON.stringify({ system, options })
  const keys = prior.map(keyOf)
  const entry = threadId ? entries.get(threadId) : undefined
  let miss: string | undefined
  if (entry) {
    miss = entry.busy
      ? "another call of this thread holds its session"
      : entry.setup !== setup
        ? "the system prompt, tools or options changed"
        : startsWith(keys, entry.history)
          ? undefined
          : "the conversation changed (an edited, regenerated or rewritten message)"
  }

  if (entry && !miss) {
    entry.busy = true
    // Most recently used last.
    entries.delete(threadId as string)
    entries.set(threadId as string, entry)
    const cachedTokens = entry.session.contextUsage
    const started = performance.now()
    const gap = prior.slice(entry.history.length)
    entry.overflowed = false
    try {
      if (gap.length > 0) {
        await entry.session.append(await toLanguageModelMessages(gap, media), {
          signal
        })
      }
      entry.history = keys
      return {
        ...lease(threadId, entry, cachedTokens, performance.now() - started),
        reused: true
      }
    } catch (error) {
      drop(threadId as string, entry)
      if (signal?.aborted) throw error
      // The session couldn't take them (it was destroyed, say): start over.
      miss = `the kept session failed (${error instanceof Error ? error.message : String(error)})`
    }
  }

  // Another run of this thread holds its session (it shouldn't happen):
  // this one gets a session of its own, freed when it ends.
  const shared = !entry?.busy
  if (entry && shared && threadId) drop(threadId, entry)
  const started = performance.now()
  const created: Entry = {
    session: await createSession(system, prior, options, signal, media),
    setup,
    history: keys,
    busy: true,
    overflowed: false
  }
  created.session.addEventListener("contextoverflow", () => {
    created.overflowed = true
  })
  if (threadId && shared) {
    entries.set(threadId, created)
    for (const [id, old] of entries) {
      if (entries.size <= MAX_SESSIONS) break
      if (!old.busy) drop(id, old)
    }
  }
  return {
    ...lease(
      threadId && shared ? threadId : undefined,
      created,
      0,
      performance.now() - started
    ),
    reused: false,
    ...(miss && { miss })
  }
}

function lease(
  threadId: string | undefined,
  entry: Entry,
  cachedTokens: number,
  createMs: number
): Omit<Lease, "reused"> {
  return {
    session: entry.session,
    cachedTokens,
    createMs,
    overflowed: () => entry.overflowed,
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

/** The conversation without its oldest turn: up to the next user message. */
function withoutOldestTurn(messages: Array<PromptMessage>) {
  const next = messages.findIndex((m, i) => i > 0 && m.role === "user")
  return next > 0 ? messages.slice(next) : []
}

/**
 * A new session that has read the system prompt and the conversation. Only
 * the system prompt is an initial prompt, which the browser never drops;
 * the conversation is appended, so when it later outgrows the context the
 * browser drops its oldest turns. If it doesn't fit now, its oldest turns
 * are left out (measured first, not tried) until it does.
 */
async function createSession(
  system: string,
  prior: Array<PromptMessage>,
  options: LanguageModelCreateOptions,
  signal: AbortSignal | undefined,
  media: Map<string, Promise<Blob>> | undefined
) {
  const session = await createLanguageModel(
    {
      ...options,
      ...(system && { initialPrompts: [{ role: "system", content: system }] })
    },
    signal
  )
  try {
    let rest = prior
    while (rest.length > 0) {
      const input = await toLanguageModelMessages(rest, media)
      const room = session.contextWindow - session.contextUsage
      const needed = await session.measureContextUsage(input, { signal })
      if (needed <= room) {
        await session.append(input, { signal })
        break
      }
      rest = withoutOldestTurn(rest)
    }
    return session
  } catch (error) {
    session.destroy()
    throw error
  }
}

/** Frees every session kept, as when the page no longer needs the model. */
export function destroyPromptApiSessions() {
  for (const [id, entry] of entries) {
    if (!entry.busy) drop(id, entry)
  }
}
