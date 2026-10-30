/**
 * useChat's connection. There is no server route: each run is TanStack AI's
 * `chat()`, here in the page, streamed to useChat with `stream()` (TanStack's
 * pattern for an in-process `chat()`). A server version would be
 * `fetchServerSentEvents` to a route running this same `chat()` with a
 * hosted model's adapter.
 */
import { chat, chatParamsFromRequestBody } from "@tanstack/ai"
import type { ChatMiddleware, StreamChunk } from "@tanstack/ai"
import { stream } from "@tanstack/ai-react"

import { getMcpClients, reportMcpDiscoveryError } from "#/lib/mcp"
import { currentSystemPrompt } from "#/lib/settings"
import { compactToolResult, shorten } from "#/lib/sources"
import {
  recordGeneration,
  startReply,
  statsFromUsage
} from "./generation-stats"
import { isReasoningLevel, selectedModel } from "./model"
import { effortFor, isModelId, modelInfo } from "./models"

/**
 * Each model call's usage, as the adapter reports it when the call ends, to
 * the stats line. A reply that calls tools makes one call per step.
 */
const replyStats: ChatMiddleware = {
  name: "reply-stats",
  onUsage: (ctx, usage) =>
    recordGeneration(
      ctx.threadId,
      `${ctx.runId}:${ctx.iteration}`,
      statsFromUsage(usage)
    )
}

/**
 * Tool results in a short form, for small models' small context windows:
 * search results as a numbered source list, cut to a few thousand
 * characters. The model and the stream (which the source chips read) both
 * get the short form.
 */
const compactToolResults: ChatMiddleware = {
  name: "compact-tool-results",
  onAfterToolCall: (_ctx, info) => {
    if (!info.ok || typeof info.result !== "string") return
    return {
      type: "replaceResult",
      result: shorten(compactToolResult(info.result))
    }
  }
}

/** The model a session forwarded with this run, if it's one we have. */
function readModel(forwarded: Record<string, unknown> | undefined) {
  const model = forwarded?.model
  return isModelId(model) ? model : selectedModel()
}

/** The effort a session forwarded with this run, high if none. */
function readEffort(forwarded: Record<string, unknown> | undefined) {
  const effort = forwarded?.effort
  return isReasoningLevel(effort) ? effort : "high"
}

/** The AG-UI request useChat makes for a run. */
type RunAgentInput = Parameters<Parameters<typeof stream>[0]>[3]

/**
 * One reply: `chat()` with the forwarded model's adapter. It calls the
 * model, runs the tools it asks for (pausing for approval where a tool
 * needs it), and calls it again with their results until it answers.
 */
async function* reply(
  input: RunAgentInput,
  signal: AbortSignal | undefined
): AsyncIterable<StreamChunk> {
  const params = await chatParamsFromRequestBody(input)
  if (params.messages.at(-1)?.role === "user") startReply(params.threadId)
  const model = readModel(params.forwardedProps)
  const reasoning = effortFor(model, readEffort(params.forwardedProps))
  // stop() aborts the run, here in the page.
  const abortController = new AbortController()
  signal?.addEventListener("abort", () => abortController.abort(), {
    once: true
  })
  yield* chat({
    adapter: modelInfo(model).adapter,
    messages: params.messages,
    threadId: params.threadId,
    runId: params.runId,
    ...(params.parentRunId && { parentRunId: params.parentRunId }),
    ...(params.resume && { resume: params.resume }),
    // The user's system prompt from Settings, then the date.
    systemPrompts: [currentSystemPrompt()],
    // The MCP servers' tools, Exa's web search first. The clients stay open
    // between runs; a server that fails to list its tools is left out.
    mcp: {
      clients: getMcpClients(),
      connection: "keep-alive",
      onDiscoveryError: reportMcpDiscoveryError
    },
    // The effort picked, clamped to the model's levels. Models that don't
    // reason get none. The type is the levels these adapters declare;
    // effortFor already picked one this model has.
    ...(reasoning && { reasoning: reasoning as "off" | "high" }),
    middleware: [replyStats, compactToolResults],
    abortController
  })
}

export const browserChatConnection = stream((_messages, _data, signal, input) =>
  reply(input, signal)
)
