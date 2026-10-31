/**
 * The conversation as the Prompt API reads it. Messages are kept in a light
 * form first, with images and audio as where to get them, so a session that
 * has read part of a conversation can be matched against it by key; only
 * the messages a session actually reads are loaded (`toLanguageModelMessages`).
 */
import type { ContentPart, ModelMessage } from "@tanstack/ai"

import { PromptApiError } from "./model.js"
import type { MediaInput } from "./model.js"
import { parseArguments, toolCallText } from "./tools.js"
import { toolClasses } from "./types.js"
import type {
  LanguageModelContent,
  LanguageModelMessage,
  LanguageModelToolCall,
  LanguageModelToolResultContent
} from "./types.js"

/** Where an image's or a clip's bytes are: base64 data or a URL. */
type MediaSource = { type: "data" | "url"; value: string; mimeType?: string }

export type PromptPart =
  | { type: "text"; value: string }
  | { type: MediaInput; source: MediaSource }
  | { type: "tool-call"; call: LanguageModelToolCall }
  | {
      type: "tool-response"
      callId: string
      name: string
      result?: string
      error?: string
    }

export type PromptMessage = {
  role: "user" | "assistant"
  content: string | Array<PromptPart>
}

function partKey(part: PromptPart) {
  switch (part.type) {
    case "text":
      return `t:${part.value}`
    case "image":
    case "audio": {
      // Long base64 stays out of the key: its length and ends tell it apart.
      const { type, value, mimeType } = part.source
      return `${part.type}:${type}:${mimeType ?? ""}:${value.length}:${value.slice(0, 64)}:${value.slice(-64)}`
    }
    case "tool-call":
      return `c:${JSON.stringify([part.call.callId, part.call.name, part.call.arguments])}`
    case "tool-response":
      return `r:${JSON.stringify([part.callId, part.name, part.result, part.error])}`
  }
}

/** A message's identity, to tell what a session has already read. */
export const keyOf = (m: PromptMessage) =>
  `${m.role}\u0000${
    typeof m.content === "string"
      ? m.content
      : m.content.map(partKey).join("\u0001")
  }`

/** The text of a message's content, without its images or audio. */
export function contentText(content: ModelMessage["content"]) {
  if (typeof content === "string") return content
  return (content ?? [])
    .map((part) => (part.type === "text" ? part.content : ""))
    .join("")
}

function userContent(
  content: ModelMessage["content"]
): PromptMessage["content"] | undefined {
  if (typeof content === "string" || !content) return content || undefined
  if (content.every((part) => part.type === "text")) {
    return contentText(content) || undefined
  }
  return content.map((part: ContentPart): PromptPart => {
    if (part.type === "text") return { type: "text", value: part.content }
    if (part.type === "image" || part.type === "audio") {
      const { source } = part
      if (source.type === "data" || source.type === "url") {
        return { type: part.type, source }
      }
      throw new PromptApiError(
        `The built-in model can't read a provider's file handle; send the ${part.type} as data or a URL.`
      )
    }
    throw new PromptApiError(
      `The built-in model reads text, images and audio, not ${part.type}.`
    )
  })
}

/**
 * The model's message: its text and tool calls. With the browser's own tool
 * use they're tool-call pieces; without it, the `{"tool", "arguments"}` JSON
 * the model wrote, as text.
 */
export function assistantMessage(
  text: string,
  calls: Array<LanguageModelToolCall>,
  nativeTools: boolean
): PromptMessage | undefined {
  if (calls.length === 0)
    return text ? { role: "assistant", content: text } : undefined
  if (nativeTools) {
    return {
      role: "assistant",
      content: [
        ...(text ? [{ type: "text" as const, value: text }] : []),
        ...calls.map((call) => ({ type: "tool-call" as const, call }))
      ]
    }
  }
  return {
    role: "assistant",
    content: [text, ...calls.map((c) => toolCallText(c.name, c.arguments))]
      .filter(Boolean)
      .join("\n")
  }
}

/** A tool's result as the model reads it, or the error a failed tool gave. */
function toolResultText(m: ModelMessage, name: string | undefined) {
  const tool = name ?? "the tool"
  if (m.error) return `${tool} failed: ${m.error}`
  const text = contentText(m.content)
  return `Result of ${tool}:\n${text && text !== "null" ? text : "nothing."}`
}

/**
 * The conversation as Prompt API messages. Every step is kept, earlier
 * turns' tool calls and results included, so a session that has read a
 * conversation can take the next request by appending only what's new.
 * With `nativeTools`, tool calls and results are the browser's tool
 * messages (a step's results together, right after its calls); without,
 * calls are the JSON the model wrote and results are user messages.
 */
export function toPromptMessages(
  messages: ReadonlyArray<ModelMessage>,
  { nativeTools }: { nativeTools: boolean }
): Array<PromptMessage> {
  const out: Array<PromptMessage> = []
  const toolNames = new Map<string, string>()
  for (const m of messages) {
    if (m.role === "tool") {
      const name = m.toolCallId ? toolNames.get(m.toolCallId) : undefined
      if (!nativeTools) {
        out.push({ role: "user", content: toolResultText(m, name) })
        continue
      }
      const part: PromptPart = {
        type: "tool-response",
        callId: m.toolCallId ?? "",
        name: name ?? "tool",
        ...(m.error ? { error: m.error } : { result: contentText(m.content) })
      }
      const last = out.at(-1)
      if (
        last?.role === "user" &&
        Array.isArray(last.content) &&
        last.content.every((p) => p.type === "tool-response")
      ) {
        last.content.push(part)
      } else {
        out.push({ role: "user", content: [part] })
      }
    } else if (m.role === "assistant") {
      const calls = (m.toolCalls ?? []).map((call) => {
        toolNames.set(call.id, call.function.name)
        return {
          callId: call.id,
          name: call.function.name,
          arguments: parseArguments(call.function.arguments) ?? {}
        }
      })
      const message = assistantMessage(
        contentText(m.content),
        calls,
        nativeTools
      )
      if (message) out.push(message)
    } else {
      const content = userContent(m.content)
      if (content) out.push({ role: "user", content })
    }
  }
  return out
}

/** The kinds of media in a conversation, which its session must expect. */
export function mediaIn(messages: ReadonlyArray<PromptMessage>) {
  const kinds = new Set<MediaInput>()
  for (const m of messages) {
    if (typeof m.content === "string") continue
    for (const part of m.content) {
      if (part.type === "image" || part.type === "audio") kinds.add(part.type)
    }
  }
  return [...kinds].sort()
}

function base64Blob(value: string, mimeType: string | undefined) {
  const data = value.startsWith("data:")
    ? value.slice(value.indexOf(",") + 1)
    : value
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0))
  return new Blob([bytes], { type: mimeType ?? "" })
}

async function loadMedia(type: MediaInput, source: MediaSource) {
  if (source.type === "data") return base64Blob(source.value, source.mimeType)
  try {
    const response = await fetch(source.value)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.blob()
  } catch (error) {
    const where = source.value.startsWith("data:") ? "data URL" : source.value
    throw new PromptApiError(
      `Couldn't load the ${type} for the built-in model (${where.slice(0, 120)}): ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

/** A tool's result for the browser: its JSON object if it is one, else text. */
function toolResultContent(result: string): LanguageModelToolResultContent {
  try {
    const value = JSON.parse(result) as unknown
    if (value && typeof value === "object") return { type: "object", value }
  } catch {
    // Plain text.
  }
  return { type: "text", value: result }
}

/**
 * Messages a session reads, with their images and audio loaded (as Blobs)
 * and tool messages made of the browser's classes. `cache` keeps loaded
 * media across calls that read the same messages.
 */
export async function toLanguageModelMessages(
  messages: ReadonlyArray<PromptMessage>,
  cache: Map<string, Promise<Blob>> = new Map()
): Promise<Array<LanguageModelMessage>> {
  const classes = toolClasses()
  const toContent = async (part: PromptPart): Promise<LanguageModelContent> => {
    switch (part.type) {
      case "text":
        return part
      case "image":
      case "audio": {
        const key = partKey(part)
        let blob = cache.get(key)
        if (!blob) {
          blob = loadMedia(part.type, part.source)
          cache.set(key, blob)
        }
        return { type: part.type, value: await blob }
      }
      case "tool-call":
        if (!classes)
          throw new PromptApiError("This browser has no tool messages.")
        return { type: "tool-call", value: new classes.ToolCall(part.call) }
      case "tool-response": {
        if (!classes)
          throw new PromptApiError("This browser has no tool messages.")
        const { callId, name } = part
        return {
          type: "tool-response",
          value:
            part.error !== undefined
              ? new classes.ToolError({
                  callId,
                  name,
                  errorMessage: part.error
                })
              : new classes.ToolSuccess({
                  callId,
                  name,
                  result: [toolResultContent(part.result ?? "")]
                })
        }
      }
    }
  }
  return Promise.all(
    messages.map(async (m) =>
      typeof m.content === "string"
        ? { role: m.role, content: m.content }
        : { role: m.role, content: await Promise.all(m.content.map(toContent)) }
    )
  )
}
