import { EventType, convertMessagesToModelMessages } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

import { recordGeneration } from "#/lib/llm/generation-stats"
import {
  JSON_TOOL_CALLS_MARKER,
  createToolCallSplitter,
  parseJsonToolCalls
} from "#/lib/llm/tool-calls"
import { currentSystemPrompt } from "#/lib/settings"

/**
 * TanStack AI connection that talks to Apple's on-device Foundation Model
 * through apfel's OpenAI-compatible server (`apfel --serve --cors`).
 * No backend involved: the browser streams straight from localhost.
 */
export const APFEL_URL = "http://127.0.0.1:11434/v1/chat/completions"
export const APFEL_MODEL = "apple-foundationmodel"

type OpenAIToolCall = {
  id: string
  type: "function"
  function: { name: string; arguments: string }
}

type OpenAIMessage =
  | { role: "system" | "user"; content: string }
  | {
      role: "assistant"
      content: string | null
      tool_calls?: Array<OpenAIToolCall>
    }
  | { role: "tool"; tool_call_id: string; content: string }

/**
 * A tool result for the model. A failed tool gives its error message, so the
 * model can see what went wrong and try again.
 */
function toolResultText(m: ModelMessage) {
  const text = contentText(m.content)
  if (m.error) return `Error: ${m.error}`
  return text && text !== "null" ? text : "The tool returned nothing."
}

function contentText(content: ModelMessage["content"]) {
  if (typeof content === "string") return content
  return (content ?? [])
    .map((part) => (part.type === "text" ? part.content : ""))
    .join("")
}

function toOpenAIMessages(
  messages: Array<UIMessage> | Array<ModelMessage>
): Array<OpenAIMessage> {
  // The date and time, and the user's system prompt from Settings.
  const out: Array<OpenAIMessage> = [
    { role: "system", content: currentSystemPrompt() }
  ]
  const all = convertMessagesToModelMessages(messages)
  // Tool calls and results from earlier turns are left out; their answers
  // stay. With them in the history, Apple's model writes its next tool call
  // as broken JSON text most of the time (4 of 12 calls worked, against 12
  // of 12 without). The current turn's calls and results are kept: the
  // model answers from them.
  const currentTurn = all.map((m) => m.role).lastIndexOf("user")
  for (const [i, m] of all.entries()) {
    const earlierToolStep =
      i < currentTurn &&
      (m.role === "tool" || (m.role === "assistant" && m.toolCalls?.length))
    const content = contentText(m.content)
    if (earlierToolStep) {
      if (m.role === "assistant" && content) {
        out.push({ role: "assistant", content })
      }
    } else if (m.role === "tool") {
      out.push({
        role: "tool",
        tool_call_id: m.toolCallId ?? "",
        content: toolResultText(m)
      })
    } else if (m.role === "assistant" && m.toolCalls?.length) {
      out.push({
        role: "assistant",
        content: content || null,
        tool_calls: m.toolCalls.map((call) => ({
          id: call.id,
          type: "function",
          function: call.function
        }))
      })
    } else if (content) {
      out.push({ role: m.role === "user" ? "user" : "assistant", content })
    }
  }
  return out
}

/** Yields the `data:` payloads of an SSE response body. */
async function* sseData(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let newline: number
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim()
      buffer = buffer.slice(newline + 1)
      if (line.startsWith("data:")) yield line.slice(5).trim()
    }
  }
}

type FinishReason = "stop" | "length" | "content_filter" | "tool_calls" | null

type OpenAIChunk = {
  choices?: Array<{
    delta?: {
      content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: FinishReason
  }>
  /** Sent in a last chunk of its own when asked for with include_usage. */
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    prompt_tokens_details?: { cached_tokens?: number }
  } | null
}

/** How many times the model is told about calls to tools that don't exist. */
const MAX_UNKNOWN_TOOL_RETRIES = 1

type StreamedCall = { id: string; name: string; arguments: string }

export const apfelConnection: ConnectConnectionAdapter = {
  async *connect(messages, _data, abortSignal, runContext) {
    const { threadId, runId } = runContext
    const messageId = `msg_${runId}`
    yield { type: EventType.RUN_STARTED, threadId, runId } as StreamChunk

    const chat = toOpenAIMessages(messages)
    // The tools useChat has (from MCP servers).
    const clientTools = runContext.clientTools ?? []
    const toolNames = new Set(clientTools.map((tool) => tool.name))
    const request = () =>
      fetch(APFEL_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: APFEL_MODEL,
          stream: true,
          // Token counts, for the stats under the composer.
          stream_options: { include_usage: true },
          messages: chat,
          ...(clientTools.length > 0 && {
            tools: clientTools.map((tool) => ({
              type: "function",
              function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters
              }
            }))
          })
        }),
        signal: abortSignal
      })

    let response: Response
    let sentAt = performance.now()
    try {
      response = await request()
    } catch (error) {
      if (abortSignal?.aborted) throw error
      yield {
        type: EventType.RUN_ERROR,
        runId,
        message:
          "Couldn't reach Apple's on-device model. Start it with `apfel --serve --cors`."
      } as StreamChunk
      return
    }
    if (!response.ok || !response.body) {
      yield {
        type: EventType.RUN_ERROR,
        runId,
        message: `apfel returned HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`
      } as StreamChunk
      return
    }

    yield {
      type: EventType.TEXT_MESSAGE_START,
      messageId,
      role: "assistant"
    } as StreamChunk

    // One completion, streamed into the reply. When the model's only tool
    // calls are to tools that don't exist, it's told so and asked once more,
    // so it answers with what it has instead of stopping there.
    let finishReason: FinishReason = null
    let calls: Array<StreamedCall> = []
    for (let attempt = 0; ; attempt++) {
      let text = ""
      // Tool calls stream in pieces, by index.
      const streamed: Array<StreamedCall> = []
      // When apfel doesn't recognize a tool call, it passes it on as text;
      // that part is held back and read once the reply ends.
      const splitter = createToolCallSplitter([JSON_TOOL_CALLS_MARKER])
      // apfel streams a few tokens per chunk: the first chunk's share of the
      // characters stands in for its share of the tokens.
      const timing = { first: 0, last: 0, firstChars: 0, chars: 0 }
      let usage: OpenAIChunk["usage"]
      for await (const data of sseData(response.body)) {
        if (data === "[DONE]") break
        const chunk = JSON.parse(data) as OpenAIChunk
        usage = chunk.usage ?? usage
        const choice = chunk.choices?.[0]
        const chars =
          (choice?.delta?.content?.length ?? 0) +
          (choice?.delta?.tool_calls ?? []).reduce(
            (n, piece) =>
              n +
              (piece.function?.name?.length ?? 0) +
              (piece.function?.arguments?.length ?? 0),
            0
          )
        if (chars > 0) {
          timing.last = performance.now()
          if (!timing.first) {
            timing.first = timing.last
            timing.firstChars = chars
            // Token counts come only with the usage at the end.
            recordGeneration(
              threadId,
              `${runId}_${attempt}`,
              { firstTokenMs: timing.first - sentAt },
              false
            )
          }
          timing.chars += chars
        }
        const delta = choice?.delta?.content
        if (delta) {
          text += delta
          const shown = splitter.push(delta)
          if (shown) {
            yield {
              type: EventType.TEXT_MESSAGE_CONTENT,
              messageId,
              delta: shown
            } as StreamChunk
          }
        }
        for (const piece of choice?.delta?.tool_calls ?? []) {
          const call = (streamed[piece.index] ??= {
            id: piece.id ?? `call_${runId}_${attempt}_${piece.index}`,
            name: "",
            arguments: ""
          })
          call.name += piece.function?.name ?? ""
          call.arguments += piece.function?.arguments ?? ""
        }
        if (choice?.finish_reason) finishReason = choice.finish_reason
      }
      const outputTokens = usage?.completion_tokens
      recordGeneration(threadId, `${runId}_${attempt}`, {
        inputTokens: usage?.prompt_tokens,
        cachedTokens: usage?.prompt_tokens_details?.cached_tokens,
        outputTokens,
        ...(timing.first && {
          firstTokenMs: timing.first - sentAt,
          ...(outputTokens !== undefined && {
            decodeTokens:
              (outputTokens * (timing.chars - timing.firstChars)) /
              timing.chars,
            decodeMs: timing.last - timing.first
          })
        })
      })
      const { text: rest, markup } = splitter.flush()
      const fromText =
        markup && streamed.length === 0
          ? parseJsonToolCalls(markup, toolNames)
          : []
      fromText.forEach((call, i) =>
        streamed.push({
          id: `call_${runId}_${attempt}_text${i}`,
          name: call.name,
          arguments: JSON.stringify(call.args)
        })
      )
      // Text that wasn't a readable tool call is shown after all.
      const unread = `${rest}${fromText.length > 0 ? "" : (markup ?? "")}`
      if (unread) {
        yield {
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId,
          delta: unread
        } as StreamChunk
      }
      calls = streamed.filter((c) => toolNames.has(c.name))
      const unknown = streamed.filter((c) => !toolNames.has(c.name))
      if (
        calls.length > 0 ||
        unknown.length === 0 ||
        attempt >= MAX_UNKNOWN_TOOL_RETRIES ||
        abortSignal?.aborted
      ) {
        break
      }
      chat.push({
        role: "assistant",
        content: text || null,
        tool_calls: unknown.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.arguments }
        }))
      })
      for (const call of unknown) {
        chat.push({
          role: "tool",
          tool_call_id: call.id,
          content: `Error: there is no tool named "${call.name}". Available tools: ${[...toolNames].join(", ")}. Answer with what you have.`
        })
      }
      try {
        sentAt = performance.now()
        response = await request()
      } catch {
        break
      }
      if (!response.ok || !response.body) break
    }

    for (const call of calls) {
      yield {
        type: EventType.TOOL_CALL_START,
        toolCallId: call.id,
        toolCallName: call.name,
        parentMessageId: messageId
      } as StreamChunk
      yield {
        type: EventType.TOOL_CALL_ARGS,
        toolCallId: call.id,
        delta: call.arguments || "{}"
      } as StreamChunk
      yield {
        type: EventType.TOOL_CALL_END,
        toolCallId: call.id
      } as StreamChunk
    }
    yield { type: EventType.TEXT_MESSAGE_END, messageId } as StreamChunk
    // A run that calls browser tools ends with them pending: useChat runs
    // them and sends the results back for the next run.
    yield (
      calls.length > 0
        ? {
            type: EventType.RUN_FINISHED,
            threadId,
            runId,
            outcome: {
              type: "success",
              pendingToolCallIds: calls.map((c) => c.id)
            },
            model: APFEL_MODEL
          }
        : {
            type: EventType.RUN_FINISHED,
            threadId,
            runId,
            finishReason: finishReason ?? "stop",
            model: APFEL_MODEL
          }
    ) as StreamChunk
  }
}
