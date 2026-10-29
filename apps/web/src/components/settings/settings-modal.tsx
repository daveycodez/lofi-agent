/**
 * Settings, in a modal. For now it holds the user's MCP servers: add one by
 * URL, see whether the browser can reach it and which tools it offers, turn
 * it off, or remove it.
 */
import ArrowRotateRight from "@gravity-ui/icons/ArrowRotateRight"
import TrashBin from "@gravity-ui/icons/TrashBin"
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
import { useEffect, useState } from "react"

import {
  addMcpServer,
  checkMcpServer,
  checkMcpServers,
  mcpServersStore,
  mcpStatusStore,
  removeMcpServer,
  setMcpServerEnabled
} from "#/lib/mcp"
import type { McpServer } from "#/lib/mcp"

export function SettingsModal({
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
            <Modal.Heading>Settings</Modal.Heading>
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
  const servers = mcpServersStore.use()

  // Check servers this session hasn't reached yet.
  useEffect(checkMcpServers, [])

  return (
    <section className="flex flex-col gap-4">
      <div>
        <h3 className="text-sm font-medium">MCP Servers</h3>
        <p className="mt-1 text-sm text-muted">
          Give the model tools from remote MCP servers. Your browser connects to
          each server directly, so the server has to allow connections from this
          site.
        </p>
      </div>
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
          <p className="truncate text-sm font-medium">{server.name}</p>
          <p className="truncate text-xs text-muted">{server.url}</p>
        </div>
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
        <Switch
          aria-label={`Use ${server.name}`}
          isSelected={server.enabled}
          onChange={(enabled) => {
            setMcpServerEnabled(server.id, enabled)
            if (enabled && status?.state !== "connected") {
              void checkMcpServer({ ...server, enabled })
            }
          }}
          className="mt-1"
        >
          <Switch.Content>
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Content>
        </Switch>
      </div>

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

/** Accepts https URLs, plus http for servers on this computer. */
function urlError(value: string) {
  try {
    const url = new URL(value.trim())
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

function AddMcpServerForm() {
  const [url, setUrl] = useState("")
  const [name, setName] = useState("")
  const [token, setToken] = useState("")

  return (
    <Form
      className="flex flex-col gap-3"
      onSubmit={(e) => {
        e.preventDefault()
        if (urlError(url)) return
        addMcpServer({ url, name, token })
        setUrl("")
        setName("")
        setToken("")
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
      <div className="grid gap-3 sm:grid-cols-2">
        <TextField variant="secondary" value={name} onChange={setName}>
          <Label>Name</Label>
          <Input placeholder="Optional" />
        </TextField>
        <TextField
          type="password"
          variant="secondary"
          value={token}
          onChange={setToken}
        >
          <Label>Access token</Label>
          <Input placeholder="Optional" autoComplete="off" />
          <Description>Kept in this browser only.</Description>
        </TextField>
      </div>
      <Button type="submit" className="self-start">
        Add Server
      </Button>
    </Form>
  )
}
