/**
 * The prompt for a chat, written by the model's own chat template (the
 * Jinja template in its GGUF file), the way Transformers renders it.
 */
import { Template } from "@huggingface/jinja"

/** A chat turn as LFM2.5's chat template takes it. */
export type ChatTurn = {
  role: "system" | "user" | "assistant" | "tool"
  /**
   * The text. An assistant turn's is the reply as the model wrote it:
   * thinking (`<think>…</think>`) and tool-call markup included. The
   * template drops the thinking of all but the last assistant turn.
   */
  content: string
}

/** A tool as chat templates take it (OpenAI's function format). */
export type ToolSpec = {
  type: "function"
  function: { name: string; description: string; parameters: unknown }
}

const templates = new Map<string, Template>()

/**
 * The prompt for `chat`, ending where the model's reply starts. The
 * template's BOS token is left out: llama.cpp adds it when it reads the
 * prompt.
 */
export function renderChat(
  source: string,
  chat: Array<ChatTurn>,
  tools: Array<ToolSpec>
) {
  let template = templates.get(source)
  if (!template) {
    template = new Template(source)
    templates.set(source, template)
  }
  return template.render({
    messages: chat,
    ...(tools.length > 0 && { tools }),
    add_generation_prompt: true,
    bos_token: "",
    eos_token: ""
  })
}
