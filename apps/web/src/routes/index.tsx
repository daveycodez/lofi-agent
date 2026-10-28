import { createFileRoute } from "@tanstack/react-router"

import { Thread } from "#/components/chat/thread"
import { ThemeSwitcher } from "#/components/theme-switcher"

export const Route = createFileRoute("/")({ component: ChatPage })

function ChatPage() {
  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center justify-between px-4 py-3">
        <span className="font-medium">lofi-agent</span>
        <ThemeSwitcher />
      </header>
      <main className="min-h-0 flex-1">
        <Thread />
      </main>
    </div>
  )
}
