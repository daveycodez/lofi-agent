import { createChatUI } from "@tanstack/ai-react/ui"

import { chatOptions } from "#/chat/options"
import { chatContext, interruptContext, partContext } from "#/chat/ui-context"
import { FallbackInterrupt } from "./fallback-interrupt"
import { FallbackPart } from "./fallback-part"
import { ChatPromptInput } from "./input"
import { ChatLayout } from "./layout"
import { everyTool, McpTool, McpToolApproval } from "./mcp-tool"
import { ChatMessage } from "./message"
import {
  AudioPart,
  DocumentPart,
  ImagePart,
  ToolResultPart,
  UIResourcePart,
  VideoPart
} from "./media-parts"
import { ChatQueueItem } from "./queue-item"
import { StructuredOutputPart } from "./structured-output"
import { TextPart } from "./text-part"
import { ThinkingPart } from "./thinking-part"

/**
 * The chat kit, bound to this app's widgets. `createChatUI` rather than
 * `createChatHook`: each session calls useChat itself (see thread.tsx), to
 * give it the definitions of the MCP tools that ask first, which are only
 * known at runtime. `createChatHook` takes tools only in `chatOptions`.
 */
export const ChatUI = createChatUI(chatOptions, {
  context: {
    chatContext,
    partContext,
    interruptContext
  },
  components: {
    // `input` before `layout`, so the layout's props type includes `Input`.
    input: ChatPromptInput,
    layout: ChatLayout,
    message: ChatMessage,
    queue: ChatQueueItem
  },
  partsComponents: {
    text: TextPart,
    thinking: ThinkingPart,
    structuredOutput: StructuredOutputPart,
    image: ImagePart,
    audio: AudioPart,
    video: VideoPart,
    document: DocumentPart,
    toolResult: ToolResultPart,
    uiResource: UIResourcePart,
    fallback: FallbackPart
  },
  // MCP tools are only known at runtime: one widget for every tool name.
  toolsComponents: everyTool(McpTool),
  interruptsComponents: {
    // Approvals show in the layout's `Interrupts` list, as in TanStack's
    // example, one widget for every tool that asks first.
    tools: everyTool(McpToolApproval),
    generic: { fallback: FallbackInterrupt }
  }
})

export const { useChatContext } = ChatUI
