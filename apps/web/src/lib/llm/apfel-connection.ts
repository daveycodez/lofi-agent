import { EventType, convertMessagesToModelMessages } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

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
  | { role: "user"; content: string }
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
  // No system prompt: just the conversation.
  const out: Array<OpenAIMessage> = []
  for (const m of convertMessagesToModelMessages(messages)) {
    const content = contentText(m.content)
    if (m.role === "tool") {
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
}

export const apfelConnection: ConnectConnectionAdapter = {
  async *connect(messages, _data, abortSignal, runContext) {
    const { threadId, runId } = runContext
    const messageId = `msg_${runId}`
    yield { type: EventType.RUN_STARTED, threadId, runId } as StreamChunk

    let response: Response
    try {
      response = await fetch(APFEL_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: APFEL_MODEL,
          stream: true,
          messages: toOpenAIMessages(messages),
          // The tools useChat has (from MCP servers).
          ...(runContext.clientTools?.length && {
            tools: runContext.clientTools.map((tool) => ({
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

    let finishReason: FinishReason = null
    // Tool calls stream in pieces, by index.
    const calls: Array<{ id: string; name: string; arguments: string }> = []
    for await (const data of sseData(response.body)) {
      if (data === "[DONE]") break
      const choice = (JSON.parse(data) as OpenAIChunk).choices?.[0]
      const delta = choice?.delta?.content
      if (delta) {
        yield {
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId,
          delta
        } as StreamChunk
      }
      for (const piece of choice?.delta?.tool_calls ?? []) {
        const call = (calls[piece.index] ??= {
          id: piece.id ?? `call_${runId}_${piece.index}`,
          name: "",
          arguments: ""
        })
        call.name += piece.function?.name ?? ""
        call.arguments += piece.function?.arguments ?? ""
      }
      if (choice?.finish_reason) finishReason = choice.finish_reason
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
