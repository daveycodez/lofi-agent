import Archive from "@gravity-ui/icons/Archive"
import Check from "@gravity-ui/icons/Check"
import Display from "@gravity-ui/icons/Display"
import Ellipsis from "@gravity-ui/icons/Ellipsis"
import LayoutSideContentLeft from "@gravity-ui/icons/LayoutSideContentLeft"
import Moon from "@gravity-ui/icons/Moon"
import Sun from "@gravity-ui/icons/Sun"
import TrashBin from "@gravity-ui/icons/TrashBin"
import { Button, Drawer, Dropdown, Header, Label, Tooltip } from "@heroui/react"
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { useTheme } from "next-themes"
import { useState } from "react"
import { v7 as uuidv7 } from "uuid"

import { Thread } from "#/components/chat/thread"
import { ThreadListSidebar } from "#/components/chat/thread-list-sidebar"
import { createStore } from "#/lib/store"
import { archiveThread, deleteThread, threadsStore } from "#/lib/threads"

type Search = { thread?: string }

export const Route = createFileRoute("/")({
  validateSearch: (search: Record<string, unknown>): Search => ({
    thread: typeof search.thread === "string" ? search.thread : undefined
  }),
  component: Assistant
})

const sidebarOpenStore = createStore(true, "lofi-agent:sidebar-open")

const isDesktop = () => window.matchMedia("(min-width: 768px)").matches

/** App shell matching assistant-ui.com's demo: 240px sidebar, 48px header. */
function Assistant() {
  const { thread } = Route.useSearch()
  const navigate = useNavigate({ from: "/" })
  // A new thread gets its id up front; it joins the URL and sidebar on first send.
  const [draftId, setDraftId] = useState(() => uuidv7())
  const threadId = thread ?? draftId

  const sidebarOpen = sidebarOpenStore.use()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const meta = threadsStore.use().find((t) => t.id === threadId)

  const newThread = () => {
    setDraftId(uuidv7())
    void navigate({ search: {} })
  }

  const sidebarProps = {
    activeId: threadId,
    onNewThread: newThread,
    onActiveRemoved: newThread
  }

  return (
    <div className="flex h-dvh w-full bg-background">
      {/* Desktop: collapsible sidebar. */}
      <aside
        aria-label="Threads"
        data-open={sidebarOpen || undefined}
        className="hidden w-0 shrink-0 overflow-hidden border-separator transition-[width] duration-200 ease-linear data-open:w-60 data-open:border-r md:block"
      >
        <div className="h-full w-60">
          <ThreadListSidebar
            {...sidebarProps}
            onCollapse={() => sidebarOpenStore.set(false)}
          />
        </div>
      </aside>

      {/* Small screens: the same sidebar in a drawer. */}
      <Drawer.Backdrop isOpen={drawerOpen} onOpenChange={setDrawerOpen}>
        <Drawer.Content placement="left" className="w-60 max-w-[85vw] p-0">
          <Drawer.Dialog
            aria-label="Threads"
            className="h-full bg-background p-0"
          >
            <ThreadListSidebar
              {...sidebarProps}
              onCollapse={() => setDrawerOpen(false)}
              onNavigate={() => setDrawerOpen(false)}
            />
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-separator px-4">
          <Tooltip delay={300}>
            <Button
              isIconOnly
              variant="ghost"
              aria-label="Open threads"
              onPress={() => {
                if (isDesktop()) sidebarOpenStore.set(true)
                else setDrawerOpen(true)
              }}
              // Shown on small screens, and on desktop once the sidebar is collapsed.
              size="sm"
              className={`-ms-2 ${sidebarOpen ? "md:hidden" : ""}`}
            >
              <LayoutSideContentLeft />
            </Button>
            <Tooltip.Content placement="bottom">Open threads</Tooltip.Content>
          </Tooltip>
          <h1 className="min-w-0 flex-1 truncate text-sm font-medium">
            {meta?.title ?? "New chat"}
          </h1>
          <HeaderMenu
            threadId={meta ? threadId : undefined}
            onRemoved={newThread}
          />
        </header>
        <main className="min-h-0 flex-1">
          <Thread
            key={threadId}
            threadId={threadId}
            onSend={() => {
              if (!thread) void navigate({ search: { thread: threadId } })
            }}
          />
        </main>
      </div>
    </div>
  )
}

/** The header's ⋯ menu: actions for the open thread, plus the theme. */
function HeaderMenu({
  threadId,
  onRemoved
}: {
  threadId?: string
  onRemoved: () => void
}) {
  const { theme, setTheme } = useTheme()
  return (
    <Dropdown>
      <Tooltip delay={300}>
        <Button
          isIconOnly
          variant="ghost"
          aria-label="More actions"
          size="sm"
          className="-me-2"
        >
          <Ellipsis />
        </Button>
        <Tooltip.Content placement="bottom">More actions</Tooltip.Content>
      </Tooltip>
      <Dropdown.Popover placement="bottom end" className="min-w-48">
        <Dropdown.Menu
          disabledKeys={threadId ? [] : ["archive", "delete"]}
          selectionMode="single"
          selectedKeys={new Set([theme ?? "system"])}
          onAction={(key) => {
            if (!threadId) return
            if (key === "archive") {
              archiveThread(threadId)
              onRemoved()
            } else if (key === "delete") {
              void deleteThread(threadId)
              onRemoved()
            }
          }}
          onSelectionChange={(keys) => {
            const [key] = keys === "all" ? [] : [...keys]
            if (key === "light" || key === "dark" || key === "system") {
              setTheme(key)
            }
          }}
        >
          {/* Checkmark on the right, like the model picker (HeroUI's menu indicator sits on the left). */}
          <Dropdown.Section>
            <Header>Theme</Header>
            <Dropdown.Item id="light" textValue="Light">
              <Sun className="size-4" />
              <Label>Light</Label>
              {(theme ?? "system") === "light" && (
                <Check aria-hidden className="ms-auto size-4 text-muted" />
              )}
            </Dropdown.Item>
            <Dropdown.Item id="dark" textValue="Dark">
              <Moon className="size-4" />
              <Label>Dark</Label>
              {(theme ?? "system") === "dark" && (
                <Check aria-hidden className="ms-auto size-4 text-muted" />
              )}
            </Dropdown.Item>
            <Dropdown.Item id="system" textValue="System">
              <Display className="size-4" />
              <Label>System</Label>
              {(theme ?? "system") === "system" && (
                <Check aria-hidden className="ms-auto size-4 text-muted" />
              )}
            </Dropdown.Item>
          </Dropdown.Section>
          <Dropdown.Section>
            <Header>Thread</Header>
            <Dropdown.Item id="archive" textValue="Archive thread">
              <Archive className="size-4" />
              <Label>Archive thread</Label>
            </Dropdown.Item>
            <Dropdown.Item
              id="delete"
              textValue="Delete thread"
              variant="danger"
            >
              <TrashBin className="size-4 text-danger" />
              <Label>Delete thread</Label>
            </Dropdown.Item>
          </Dropdown.Section>
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown>
  )
}
