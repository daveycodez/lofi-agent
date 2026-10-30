import { chat, chatParamsFromRequestBody } from "@tanstack/ai"
import type { ChatMiddleware, StreamChunk } from "@tanstack/ai"
import { stream } from "@tanstack/ai-react"

import {
  recordGeneration,
  startReply,
  statsFromUsage
} from "#/lib/llm/generation-stats"
import {
  canThink,
  currentModel,
  modelInfo,
  thinkingStore
} from "#/lib/llm/models"
import { getMcpTools } from "#/lib/mcp"
import { currentSystemPrompt } from "#/lib/settings"

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

/** The AG-UI request useChat makes for a run. */
type RunAgentInput = Parameters<Parameters<typeof stream>[0]>[3]

/**
 * One reply: TanStack AI's `chat()` with the picked model's adapter, run
 * here in the page. It calls the model, runs the tools it asks for, and
 * calls it again with their results until it answers.
 */
async function* reply(
  input: RunAgentInput,
  signal: AbortSignal | undefined
): AsyncIterable<StreamChunk> {
  const params = await chatParamsFromRequestBody(input)
  if (params.messages.at(-1)?.role === "user") startReply(params.threadId)
  const model = currentModel()
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
    // The MCP servers' tools, Exa's web search first.
    tools: getMcpTools(),
    // Think: the adapter reads any level but "off" as "think first". Models
    // without a thinking switch get none.
    ...(canThink(model) && {
      reasoning: thinkingStore.get() ? "high" : "off"
    }),
    middleware: [replyStats],
    abortController
  })
}

/**
 * useChat's connection: replies run in the browser. A server version would
 * be `fetchServerSentEvents` to a route running the same `chat()` with a
 * hosted model's adapter; nothing else changes.
 */
export const browserChatConnection = stream((_messages, _data, signal, input) =>
  reply(input, signal)
)
