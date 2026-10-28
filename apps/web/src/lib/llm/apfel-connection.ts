import { EventType } from "@tanstack/ai"
import type { ModelMessage, StreamChunk, UIMessage } from "@tanstack/ai"
import type { ConnectConnectionAdapter } from "@tanstack/ai-react"

/**
 * TanStack AI connection that talks to Apple's on-device Foundation Model
 * through apfel's OpenAI-compatible server (`apfel --serve --cors`).
 * No backend involved: the browser streams straight from localhost.
 */
export const APFEL_URL = "http://127.0.0.1:11434/v1/chat/completions"
export const APFEL_MODEL = "apple-foundationmodel"

type OpenAIMessage = { role: "system" | "user" | "assistant"; content: string }

const SYSTEM_PROMPT =
  "You are a helpful assistant running fully on the user's device. Be concise. Use Markdown when it helps."

function textOf(message: UIMessage | ModelMessage): string {
  if ("parts" in message) {
    return message.parts
      .map((part) => (part.type === "text" ? part.content : ""))
      .join("")
  }
  return typeof message.content === "string" ? message.content : ""
}

function toOpenAIMessages(
  messages: Array<UIMessage> | Array<ModelMessage>
): Array<OpenAIMessage> {
  const out: Array<OpenAIMessage> = [{ role: "system", content: SYSTEM_PROMPT }]
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue
    const content = textOf(message)
    if (content) out.push({ role: message.role, content })
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
    delta?: { content?: string | null }
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
          messages: toOpenAIMessages(messages)
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
      if (choice?.finish_reason) finishReason = choice.finish_reason
    }

    yield { type: EventType.TEXT_MESSAGE_END, messageId } as StreamChunk
    yield {
      type: EventType.RUN_FINISHED,
      threadId,
      runId,
      finishReason: finishReason ?? "stop",
      model: APFEL_MODEL
    } as StreamChunk
  }
}
