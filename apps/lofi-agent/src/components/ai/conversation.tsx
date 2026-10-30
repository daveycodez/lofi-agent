import ArrowDown from "@gravity-ui/icons/ArrowDown"
import { Spinner } from "@heroui/react"
import type { ComponentProps, ReactNode } from "react"
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom"

import { Action } from "#/components/ai/actions"

export type ConversationProps = ComponentProps<typeof StickToBottom>

/** The transcript, which follows new content while you're at the bottom. */
export function Conversation({ className = "", ...props }: ConversationProps) {
  return (
    <StickToBottom
      className={`relative flex h-full flex-col ${className}`}
      initial="instant"
      resize="smooth"
      role="log"
      {...props}
    />
  )
}

export type ConversationContentProps = ComponentProps<
  typeof StickToBottom.Content
>

export function ConversationContent({
  className = "",
  scrollClassName = "",
  ...props
}: ConversationContentProps) {
  return (
    <StickToBottom.Content
      scrollClassName={`px-6 pt-6 ${scrollClassName}`}
      className={`mx-auto flex w-full max-w-[714px] flex-col px-4 ${className}`}
      {...props}
    />
  )
}

/** An empty conversation: a heading, with whatever starts one under it. */
export function ConversationEmptyState({
  title,
  children
}: {
  title: string
  children?: ReactNode
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 pb-16">
      <div className="flex w-full max-w-[714px] flex-col items-center">
        <h2 className="mb-8 text-center text-2xl font-semibold">{title}</h2>
        {children}
      </div>
    </div>
  )
}

/**
 * Jumps back to the newest message; hidden while already there. Shows a
 * spinner instead of the arrow while a reply is being written.
 */
export function ConversationScrollButton({
  isLoading = false
}: {
  isLoading?: boolean
}) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext()
  return (
    <div
      className={`absolute -top-12 self-center ${isAtBottom ? "invisible" : ""}`}
    >
      <Action
        label="Scroll to bottom"
        variant="outline"
        // An opaque page-colored fill, so text scrolling under it doesn't
        // show through, including on hover.
        className="[--button-bg-hover:color-mix(in_srgb,var(--default)_60%,var(--background))] [--button-bg:var(--background)]"
        onPress={() => void scrollToBottom()}
      >
        {isLoading ? <Spinner size="sm" color="current" /> : <ArrowDown />}
      </Action>
    </div>
  )
}
