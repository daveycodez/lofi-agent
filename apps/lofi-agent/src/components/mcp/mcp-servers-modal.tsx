/**
 * The MCP servers modal: the built-in Exa server, then the user's own. Add
 * one by URL, see whether the browser can reach it and which tools it
 * offers, turn it off, or remove it. Built-in servers can't be removed, and
 * take an optional key.
 */
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import Plus from "@gravity-ui/icons/Plus"
import TrashBin from "@gravity-ui/icons/TrashBin"
import Xmark from "@gravity-ui/icons/Xmark"
import {
  Button,
  Chip,
  Description,
  FieldError,
  Form,
  Input,
  Label,
  Modal,
  Spinner,
  Switch,
  TextField,
  Tooltip
} from "@heroui/react"
import { useEffect, useRef, useState } from "react"

import {
  addMcpServer,
  checkMcpServer,
  checkMcpServers,
  hasMcpServer,
  mcpStatusStore,
  removeMcpServer,
  setBuiltInMcpToken,
  setMcpServerEnabled,
  useMcpServers
} from "#/lib/mcp"
import type { McpServer } from "#/lib/mcp"

export function McpServersModal({
  isOpen,
  onOpenChange
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
}) {
  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange}>
      <Modal.Container size="lg" scroll="inside">
        <Modal.Dialog>
          <Modal.CloseTrigger />
          <Modal.Header>
            <Modal.Heading>MCP Servers</Modal.Heading>
          </Modal.Header>
          <Modal.Body>
            <McpServersSection />
          </Modal.Body>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  )
}

function McpServersSection() {
  const servers = useMcpServers()

  // Check servers this session hasn't reached yet.
  useEffect(checkMcpServers, [])

  return (
    <section className="flex flex-col gap-4">
      <p className="text-sm text-muted">
        Give the model tools from remote MCP servers. Exa&apos;s web search
        comes built in. Your browser connects to each server directly, so the
        server has to allow connections from this site.
      </p>
      {servers.length > 0 && (
        <ul className="flex flex-col gap-2">
          {servers.map((server) => (
            <McpServerRow key={server.id} server={server} />
          ))}
        </ul>
      )}
      <AddMcpServerForm />
    </section>
  )
}

function McpServerRow({ server }: { server: McpServer }) {
  const status = mcpStatusStore.use()[server.id]
  return (
    <li className="flex flex-col gap-2 rounded-2xl bg-surface-secondary p-3">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-foreground">
            {server.name}
          </p>
          <p className="truncate text-xs text-muted">{server.url}</p>
        </div>
        {!server.builtIn && (
          <Tooltip delay={300}>
            <Button
              isIconOnly
              size="sm"
              variant="ghost"
              aria-label={`Remove ${server.name}`}
              onPress={() => removeMcpServer(server.id)}
            >
              <TrashBin className="text-danger" />
            </Button>
            <Tooltip.Content>Remove server</Tooltip.Content>
          </Tooltip>
        )}
        <Switch
          aria-label={`Use ${server.name}`}
          isSelected={server.enabled}
          onChange={(enabled) => {
            setMcpServerEnabled(server.id, enabled)
            if (enabled && status?.state !== "connected") {
              void checkMcpServer({ ...server, enabled })
            }
          }}
          // The default off track (--default) matches the row's surface.
          className="mt-1 [--switch-control-bg:var(--muted)]"
        >
          <Switch.Content>
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Content>
        </Switch>
      </div>

      {server.builtIn && server.tokenLabel && (
        <BuiltInKeyField server={server} label={server.tokenLabel} />
      )}
      {server.enabled && status?.state === "checking" && (
        <p className="flex items-center gap-2 text-xs text-muted">
          <Spinner size="sm" color="current" />
          Connecting…
        </p>
      )}
      {server.enabled && status?.state === "error" && (
        <div className="flex items-start gap-2">
          <p className="min-w-0 flex-1 text-xs text-danger">{status.message}</p>
          <Button
            size="sm"
            variant="ghost"
            className="-my-1.5 shrink-0"
            onPress={() => void checkMcpServer(server)}
          >
            <ArrowRotateRight />
            Retry
          </Button>
        </div>
      )}
      {server.enabled && status?.state === "connected" && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs text-muted">
            {status.tools.length === 1
              ? "1 tool"
              : `${status.tools.length} tools`}
          </p>
          {status.tools.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {status.tools.map((tool) => (
                <Chip key={tool.name} size="sm">
                  <Chip.Label title={tool.description}>{tool.name}</Chip.Label>
                </Chip>
              ))}
            </div>
          )}
        </div>
      )}
    </li>
  )
}

/**
 * The optional key of a built-in server, saved when the field loses focus
 * or on Enter.
 */
function BuiltInKeyField({
  server,
  label
}: {
  server: McpServer
  label: string
}) {
  const [value, setValue] = useState(server.token ?? "")
  const save = () => {
    if (value.trim() !== (server.token ?? "")) {
      setBuiltInMcpToken(server.id, value)
    }
  }
  return (
    <TextField
      type="password"
      value={value}
      onChange={setValue}
      onBlur={save}
      onKeyDown={(e) => {
        if (e.key === "Enter") save()
      }}
    >
      <Label>{label}</Label>
      <Input placeholder="Optional" autoComplete="off" />
      <Description>
        Works without one at a free rate limit. Kept in this browser only.
      </Description>
    </TextField>
  )
}

/** Accepts https URLs, plus http for servers on this computer. */
function urlError(value: string) {
  try {
    const url = new URL(value.trim())
    if (hasMcpServer(value)) return "This server is already in the list."
    if (url.protocol === "https:") return null
    if (
      url.protocol === "http:" &&
      ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
    ) {
      return null
    }
    return "Use an https:// URL."
  } catch {
    return "Enter a full URL, like https://example.com/mcp."
  }
}

type HeaderRow = { id: number; name: string; value: string }

/** A header name a browser can send: letters, digits and a few symbols. */
const headerNameError = (value: string) =>
  !value.trim() || /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(value.trim())
    ? null
    : "Letters, digits and dashes only."

function AddMcpServerForm() {
  const [url, setUrl] = useState("")
  const [name, setName] = useState("")
  const [headers, setHeaders] = useState<Array<HeaderRow>>([])
  const nextId = useRef(0)
  const update = (id: number, change: Partial<HeaderRow>) =>
    setHeaders((rows) =>
      rows.map((row) => (row.id === id ? { ...row, ...change } : row))
    )

  return (
    <Form
      className="flex flex-col gap-3"
      onSubmit={(e) => {
        e.preventDefault()
        if (urlError(url)) return
        if (headers.some((h) => headerNameError(h.name))) return
        addMcpServer({ url, name, headers })
        setUrl("")
        setName("")
        setHeaders([])
      }}
    >
      <TextField
        isRequired
        type="url"
        variant="secondary"
        value={url}
        onChange={setUrl}
        validate={(value) => (value ? urlError(value) : null)}
      >
        <Label>Server URL</Label>
        <Input placeholder="https://example.com/mcp" />
        <FieldError />
      </TextField>
      <TextField variant="secondary" value={name} onChange={setName}>
        <Label>Name</Label>
        <Input placeholder="Optional" />
      </TextField>
      <div className="flex flex-col gap-2">
        <div>
          <p className="text-sm font-medium text-foreground">Headers</p>
          <p className="text-xs text-muted">
            Optional, for servers that take a key, like{" "}
            <code>Authorization: Bearer …</code>. Sent with every request and
            kept in this browser only.
          </p>
        </div>
        {headers.map((row) => (
          <div key={row.id} className="flex items-start gap-2">
            <TextField
              aria-label="Header name"
              variant="secondary"
              value={row.name}
              onChange={(value) => update(row.id, { name: value })}
              validate={headerNameError}
              className="min-w-0 flex-1"
            >
              <Input placeholder="Name" autoComplete="off" />
              <FieldError />
            </TextField>
            <TextField
              aria-label="Header value"
              type="password"
              variant="secondary"
              value={row.value}
              onChange={(value) => update(row.id, { value })}
              className="min-w-0 flex-1"
            >
              <Input placeholder="Value" autoComplete="off" />
            </TextField>
            <Tooltip delay={300}>
              <Button
                isIconOnly
                variant="ghost"
                aria-label="Remove header"
                onPress={() =>
                  setHeaders((rows) => rows.filter((r) => r.id !== row.id))
                }
              >
                <Xmark />
              </Button>
              <Tooltip.Content>Remove header</Tooltip.Content>
            </Tooltip>
          </div>
        ))}
        <Button
          size="sm"
          variant="ghost"
          className="self-start"
          onPress={() =>
            setHeaders((rows) => [
              ...rows,
              { id: nextId.current++, name: "", value: "" }
            ])
          }
        >
          <Plus />
          Add header
        </Button>
      </div>
      <Button type="submit" className="self-start">
        Add Server
      </Button>
    </Form>
  )
}
