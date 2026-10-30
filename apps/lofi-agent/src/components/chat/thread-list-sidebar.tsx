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
import LogoGithub from "@gravity-ui/icons/LogoGithub"
import Pencil from "@gravity-ui/icons/Pencil"
import Pin from "@gravity-ui/icons/Pin"
import PlugConnection from "@gravity-ui/icons/PlugConnection"
import PinSlash from "@gravity-ui/icons/PinSlash"
import Plus from "@gravity-ui/icons/Plus"
import TrashBin from "@gravity-ui/icons/TrashBin"
import Xmark from "@gravity-ui/icons/Xmark"
import {
  AlertDialog,
  Button,
  buttonVariants,
  Dropdown,
  InputGroup,
  Label,
  SearchField,
  Skeleton,
  TextField
} from "@heroui/react"
import { eq, useLiveQuery, useOptionalDbClient } from "@tanstack/react-db"
import { Link } from "@tanstack/react-router"
import { Fragment, useEffect, useRef, useState } from "react"

import { threadsCollection } from "#/lib/db"
import type { Thread } from "#/lib/db"
import {
  archiveThread,
  deleteAllThreads,
  deleteThread,
  renameThread,
  togglePinThread
} from "#/lib/threads"

type SidebarProps = {
  activeId: string
  onNewThread: () => void
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
        <ChatsMenu onActiveRemoved={props.onActiveRemoved} />
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

/** Actions for all chats at once, in the header: deleting them all. */
function ChatsMenu({ onActiveRemoved }: { onActiveRemoved: () => void }) {
  // Every session, archived ones included.
  const db = useOptionalDbClient()
  const { data: allThreads } = useLiveQuery({
    query: (q) => (db ? q.from({ thread: threadsCollection }) : null)
  })
  const count = allThreads?.length ?? 0
  const [isConfirming, setIsConfirming] = useState(false)
  return (
    <>
      <Dropdown>
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          aria-label="Session options"
          className="-me-2 ms-auto"
        >
          <Ellipsis />
        </Button>
        <Dropdown.Popover placement="bottom end">
          <Dropdown.Menu
            disabledKeys={count === 0 ? ["delete-all"] : []}
            onAction={(key) => {
              if (key === "delete-all") setIsConfirming(true)
            }}
          >
            <Dropdown.Item
              id="delete-all"
              textValue="Delete all sessions"
              variant="danger"
            >
              <TrashBin className="size-4 text-danger" />
              <Label>Delete all sessions</Label>
            </Dropdown.Item>
          </Dropdown.Menu>
        </Dropdown.Popover>
      </Dropdown>

      <AlertDialog.Backdrop
        isOpen={isConfirming}
        onOpenChange={setIsConfirming}
      >
        <AlertDialog.Container>
          <AlertDialog.Dialog className="sm:max-w-[400px]">
            <AlertDialog.CloseTrigger />
            <AlertDialog.Header>
              <AlertDialog.Icon status="danger">
                <TrashBin className="size-5" />
              </AlertDialog.Icon>
              <AlertDialog.Heading>Delete all sessions?</AlertDialog.Heading>
            </AlertDialog.Header>
            <AlertDialog.Body>
              <p>
                This permanently deletes{" "}
                {count === 1 ? "your session" : `all ${count} sessions`},
                including archived ones. This can't be undone.
              </p>
            </AlertDialog.Body>
            <AlertDialog.Footer>
              <Button slot="close" variant="tertiary">
                Cancel
              </Button>
              <Button
                slot="close"
                variant="danger"
                onPress={() => {
                  deleteAllThreads()
                  onActiveRemoved()
                }}
              >
                Delete All
              </Button>
            </AlertDialog.Footer>
          </AlertDialog.Dialog>
        </AlertDialog.Container>
      </AlertDialog.Backdrop>
    </>
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
  // Off until the database is open (and on the server): the list shows
  // skeleton rows meanwhile.
  const db = useOptionalDbClient()
  const { data: threads } = useLiveQuery({
    query: (q) =>
      db
        ? q
            .from({ thread: threadsCollection })
            .where(({ thread }) => eq(thread.archived, false))
            .orderBy(({ thread }) => thread.updatedAt, "desc")
        : null
  })
  const [search, setSearch] = useState("")
  const query = search.trim().toLowerCase()
  const all = threads ?? []
  const filtered = query
    ? all.filter((t) => t.title.toLowerCase().includes(query))
    : all

  // Pinned threads first, then the rest grouped by date.
  const groups: Array<{ label: string; items: Array<Thread> }> = []
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
        New session
      </Button>

      <SearchField
        fullWidth
        aria-label="Search sessions"
        value={search}
        onChange={setSearch}
        className="mt-2 shrink-0"
      >
        <SearchField.Group>
          <SearchField.SearchIcon />
          <SearchField.Input placeholder="Search sessions" />
          <SearchField.ClearButton />
        </SearchField.Group>
      </SearchField>

      <nav aria-label="Sessions" className="mt-6 flex flex-col gap-0.5">
        {!threads ? (
          <ThreadListSkeleton />
        ) : query && filtered.length === 0 ? (
          <p className="px-2 py-2 text-sm text-muted">No sessions found</p>
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

/** Rows the size of the list's, while the sessions load. */
function ThreadListSkeleton() {
  const widths = ["w-3/4", "w-1/2", "w-2/3", "w-3/5", "w-1/2", "w-2/3"]
  return (
    <div role="status" className="flex flex-col gap-0.5">
      <span className="sr-only">Loading sessions</span>
      <Skeleton className="mb-1 ms-2 h-4 w-12 rounded-md" />
      {widths.map((width, i) => (
        <div key={i} className="flex h-10 items-center px-4 md:h-9">
          <Skeleton className={`h-3.5 rounded-md ${width}`} />
        </div>
      ))}
    </div>
  )
}

function ThreadListItem({
  thread,
  isActive,
  onNavigate,
  onActiveRemoved
}: {
  thread: Thread
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
      <TextField fullWidth aria-label="Rename session">
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
    // Each chat is a link drawn as a HeroUI button: tertiary when open, ghost
    // otherwise. The actions button sits over its end.
    <div className="group relative">
      <Link
        to="/sessions/$sessionId"
        params={{ sessionId: thread.id }}
        onClick={onNavigate}
        className={buttonVariants({
          variant: isActive ? "tertiary" : "ghost",
          fullWidth: true,
          className:
            "justify-start pe-10 font-normal group-has-aria-expanded:bg-default"
        })}
      >
        <span className="min-w-0 flex-1 truncate">
          {thread.title || "New session"}
        </span>
      </Link>
      <Dropdown>
        <Button
          isIconOnly
          size="sm"
          variant="ghost"
          aria-label="Session actions"
          className="absolute end-1 top-1/2 z-10 -translate-y-1/2"
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
                deleteThread(thread.id)
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
