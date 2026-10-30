import { createChatHook } from "@tanstack/ai-react/ui"

import { chatOptions } from "#/chat/options"
import { chatContext, interruptContext, partContext } from "#/chat/ui-context"
import { FallbackPart } from "./fallback-part"
import { ChatPromptInput } from "./input"
import { ChatLayout } from "./layout"
import { everyTool, McpTool } from "./mcp-tool"
import { ChatMessage } from "./message"
import { TextPart } from "./text-part"
import { ThinkingPart } from "./thinking-part"

export const { useAppChat, useChatContext } = createChatHook({
  options: chatOptions,
  context: {
    chatContext,
    partContext,
    interruptContext
  },
  components: {
    // `input` before `layout`, so the layout's props type includes `Input`.
    input: ChatPromptInput,
    layout: ChatLayout,
    message: ChatMessage
  },
  partsComponents: {
    text: TextPart,
    thinking: ThinkingPart,
    // A tool result shows with its call, in the tool's widget.
    toolResult: FallbackPart,
    fallback: FallbackPart
  },
  // MCP tools are only known at runtime: one widget for every tool name.
  toolsComponents: everyTool(McpTool)
})
