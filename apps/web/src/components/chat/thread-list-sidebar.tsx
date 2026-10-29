/**
 * Thread list sidebar. The layout follows assistant-ui's thread list sidebar
 * (MIT, github.com/assistant-ui/assistant-ui); everything visual is HeroUI's
 * default theme and components.
 */
import Archive from "@gravity-ui/icons/Archive"
import Check from "@gravity-ui/icons/Check"
import Comments from "@gravity-ui/icons/Comments"
import Ellipsis from "@gravity-ui/icons/Ellipsis"
import Gear from "@gravity-ui/icons/Gear"
import LayoutSideContentLeft from "@gravity-ui/icons/LayoutSideContentLeft"
import LogoGithub from "@gravity-ui/icons/LogoGithub"
import Pencil from "@gravity-ui/icons/Pencil"
import Pin from "@gravity-ui/icons/Pin"
import PlugConnection from "@gravity-ui/icons/PlugConnection"
import PinSlash from "@gravity-ui/icons/PinSlash"
import Plus from "@gravity-ui/icons/Plus"
import TrashBin from "@gravity-ui/icons/TrashBin"
import Xmark from "@gravity-ui/icons/Xmark"
import {
  Button,
  buttonVariants,
  Dropdown,
  InputGroup,
  Label,
  SearchField,
  TextField,
  Tooltip
} from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { Fragment, useEffect, useRef, useState } from "react"

import {
  archiveThread,
  deleteThread,
  renameThread,
  threadsStore,
  togglePinThread
} from "#/lib/threads"
import type { ThreadMeta } from "#/lib/threads"

type SidebarProps = {
  activeId: string
  onNewThread: () => void
  /** Collapses the desktop sidebar or closes the mobile drawer. */
  onCollapse: () => void
  /** Called after navigating (closes the mobile drawer). */
  onNavigate?: () => void
  /** Called when the active thread is archived or deleted. */
  onActiveRemoved: () => void
  onOpenMcpServers: () => void
  onOpenSettings: () => void
}

export function ThreadListSidebar(props: SidebarProps) {
  return (
    <div className="flex h-full w-full flex-col">
      {/* Header row, same height as the main header. */}
      <div className="flex h-12 shrink-0 items-center gap-2 px-4">
        <Comments className="size-4" />
        <span className="text-sm font-semibold">lofi-agent</span>
        <Tooltip delay={300}>
          <Button
            isIconOnly
            size="sm"
            variant="ghost"
            aria-label="Collapse chats"
            onPress={props.onCollapse}
            className="ms-auto -me-2"
          >
            <LayoutSideContentLeft />
          </Button>
          <Tooltip.Content>Collapse chats</Tooltip.Content>
        </Tooltip>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-3">
        <ThreadList {...props} />
      </div>

      <div className="flex flex-col gap-0.5 border-t border-separator p-3">
        <Button
          fullWidth
          variant="ghost"
          className="justify-start"
          onPress={props.onOpenMcpServers}
        >
          <PlugConnection className="text-muted" />
          MCP Servers
        </Button>
        <Button
          fullWidth
          variant="ghost"
          className="justify-start"
          onPress={props.onOpenSettings}
        >
          <Gear className="text-muted" />
          Settings
        </Button>
        {/* The source on GitHub: a link styled as the ghost buttons above. */}
        <a
          href="https://github.com/daveycodez/lofi-agent"
          target="_blank"
          rel="noopener noreferrer"
          className={buttonVariants({
            variant: "ghost",
            fullWidth: true,
            className: "justify-start"
          })}
        >
          <LogoGithub className="text-muted" />
          GitHub
        </a>
      </div>
    </div>
  )
}

function dateGroupLabel(updatedAt: number) {
  const now = new Date()
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate()
  ).getTime()
  if (updatedAt >= startOfToday) return "Today"
  if (updatedAt >= startOfToday - 86_400_000) return "Yesterday"
  return "Earlier"
}

function ThreadList({
  activeId,
  onNewThread,
  onNavigate,
  onActiveRemoved
}: SidebarProps) {
  const threads = threadsStore
    .use()
    .filter((t) => !t.archived)
    .sort((a, b) => b.updatedAt - a.updatedAt)
  const [search, setSearch] = useState("")
  const query = search.trim().toLowerCase()
  const filtered = query
    ? threads.filter((t) => t.title.toLowerCase().includes(query))
    : threads

  // Pinned threads first, then the rest grouped by date.
  const groups: Array<{ label: string; items: Array<ThreadMeta> }> = []
  const pinned = filtered.filter((t) => t.pinned)
  if (pinned.length > 0) groups.push({ label: "Pinned", items: pinned })
  for (const thread of filtered.filter((t) => !t.pinned)) {
    const label = dateGroupLabel(thread.updatedAt)
    const group = groups.at(-1)
    if (group?.label === label) group.items.push(thread)
    else groups.push({ label, items: [thread] })
  }

  return (
    <>
      {/* shrink-0: the list scrolls; these keep their size above it. */}
      <Button
        fullWidth
        variant="outline"
        className="shrink-0 justify-start"
        onPress={() => {
          onNewThread()
          onNavigate?.()
        }}
      >
        <Plus className="text-muted" />
        New chat
      </Button>

      <SearchField
        fullWidth
        aria-label="Search chats"
        value={search}
        onChange={setSearch}
        className="mt-2 shrink-0"
      >
        <SearchField.Group>
          <SearchField.SearchIcon />
          <SearchField.Input placeholder="Search chats" />
          <SearchField.ClearButton />
        </SearchField.Group>
      </SearchField>

      <nav aria-label="Chats" className="mt-6 flex flex-col gap-0.5">
        {query && filtered.length === 0 ? (
          <p className="px-2 py-2 text-sm text-muted">No chats found</p>
        ) : (
          groups.map((group, i) => (
            <Fragment key={group.label}>
              <p
                className={`mb-1 px-2 text-xs font-medium text-muted ${i > 0 ? "mt-4" : ""}`}
              >
                {group.label}
              </p>
              {group.items.map((thread) => (
                <ThreadListItem
                  key={thread.id}
                  thread={thread}
                  isActive={thread.id === activeId}
                  onNavigate={onNavigate}
                  onActiveRemoved={onActiveRemoved}
                />
              ))}
            </Fragment>
          ))
        )}
      </nav>
    </>
  )
}

function ThreadListItem({
  thread,
  isActive,
  onNavigate,
  onActiveRemoved
}: {
  thread: ThreadMeta
  isActive: boolean
  onNavigate?: () => void
  onActiveRemoved: () => void
}) {
  const [isRenaming, setIsRenaming] = useState(false)
  const [draft, setDraft] = useState(thread.title)
  const renameRef = useRef<HTMLInputElement>(null)

  // The menu returns focus to its trigger when it closes, after autoFocus has
  // run, so move focus to the rename field once that has happened.
  useEffect(() => {
    if (!isRenaming) return
    const t = setTimeout(() => {
      renameRef.current?.focus()
      renameRef.current?.select()
    }, 50)
    return () => clearTimeout(t)
  }, [isRenaming])

  if (isRenaming) {
    const save = () => {
      renameThread(thread.id, draft)
      setIsRenaming(false)
    }
    const cancel = () => {
      setDraft(thread.title)
      setIsRenaming(false)
    }
    // Same footprint as the row: the title becomes an input with save/cancel inside.
    return (
      <TextField fullWidth aria-label="Rename chat">
        <InputGroup
          fullWidth
          onBlur={(e) => {
            // Leaving the field saves, unless focus moved to its own buttons.
            if (!e.currentTarget.contains(e.relatedTarget)) save()
          }}
        >
          <InputGroup.Input
            ref={renameRef}
            className="min-w-0"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save()
              if (e.key === "Escape") cancel()
            }}
          />
          <InputGroup.Suffix className="gap-0.5 pe-1">
            <Button
              isIconOnly
              size="sm"
              variant="ghost"
              aria-label="Save name"
              onPress={save}
            >
              <Check />
            </Button>
            <Button
              isIconOnly
              size="sm"
              variant="ghost"
              aria-label="Cancel rename"
              onPress={cancel}
            >
              <Xmark />
            </Button>
          </InputGroup.Suffix>
        </InputGroup>
      </TextField>
    )
  }

  return (
    // Active row: surface + shadow; other rows: default fill on hover (HeroUI guidelines).
    <div
      data-active={isActive || undefined}
      className="group relative flex h-9 items-center rounded-lg hover:bg-default has-aria-expanded:bg-default data-active:bg-surface data-active:shadow-surface"
    >
      <Link
        to="/"
        search={{ thread: thread.id }}
        onClick={onNavigate}
        className="flex h-full min-w-0 flex-1 items-center rounded-lg px-2 text-sm outline-none group-hover:pe-9 group-has-aria-expanded:pe-9 group-data-active:pe-9 focus-visible:ring-2 focus-visible:ring-focus"
      >
        <span className="min-w-0 flex-1 truncate">
          {thread.title || "New chat"}
        </span>
      </Link>
      <Dropdown>
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          aria-label="Chat actions"
          className="absolute end-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 group-data-active:opacity-100 aria-expanded:opacity-100 focus-visible:opacity-100"
        >
          <Ellipsis />
        </Button>
        <Dropdown.Popover placement="right top">
          <Dropdown.Menu
            onAction={(key) => {
              if (key === "rename") {
                setDraft(thread.title)
                setIsRenaming(true)
              } else if (key === "pin") {
                togglePinThread(thread.id)
              } else if (key === "archive") {
                archiveThread(thread.id)
                if (isActive) onActiveRemoved()
              } else if (key === "delete") {
                void deleteThread(thread.id)
                if (isActive) onActiveRemoved()
              }
            }}
          >
            <Dropdown.Item id="rename" textValue="Rename">
              <Pencil className="size-4" />
              <Label>Rename</Label>
            </Dropdown.Item>
            <Dropdown.Item id="pin" textValue={thread.pinned ? "Unpin" : "Pin"}>
              {thread.pinned ? (
                <PinSlash className="size-4" />
              ) : (
                <Pin className="size-4" />
              )}
              <Label>{thread.pinned ? "Unpin" : "Pin"}</Label>
            </Dropdown.Item>
            <Dropdown.Item id="archive" textValue="Archive">
              <Archive className="size-4" />
              <Label>Archive</Label>
            </Dropdown.Item>
            <Dropdown.Item id="delete" textValue="Delete" variant="danger">
              <TrashBin className="size-4 text-danger" />
              <Label>Delete</Label>
            </Dropdown.Item>
          </Dropdown.Menu>
        </Dropdown.Popover>
      </Dropdown>
    </div>
  )
}
