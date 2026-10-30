import Check from "@gravity-ui/icons/Check"
import Hand from "@gravity-ui/icons/Hand"
import TriangleExclamation from "@gravity-ui/icons/TriangleExclamation"
import Wrench from "@gravity-ui/icons/Wrench"
import { Button, Card, Disclosure, Spinner } from "@heroui/react"
import type { ReactNode } from "react"

/** A tool call, in a card, collapsed under its name. */
export function Tool({ children }: { children: ReactNode }) {
  return (
    <Card className="gap-0 p-0">
      <Disclosure>{children}</Disclosure>
    </Card>
  )
}

/**
 * Where a call is: `preparing` while the model writes its arguments,
 * `running` while the tool runs, `approval` while it waits for the user,
 * `stopped` when the reply ended before it did.
 */
export type ToolStatus =
  | "preparing"
  | "running"
  | "approval"
  | "error"
  | "done"
  | "stopped"

const STATUS_LABELS: Record<ToolStatus, string> = {
  preparing: "Preparing",
  running: "Running",
  approval: "Waiting for approval",
  error: "Failed",
  done: "Done",
  stopped: "Not run"
}

function StatusIcon({ status }: { status: ToolStatus }) {
  if (status === "preparing" || status === "running") {
    return <Spinner size="sm" color="current" />
  }
  if (status === "approval") return <Hand className="text-warning" />
  if (status === "error") return <TriangleExclamation className="text-danger" />
  if (status === "done") return <Check className="text-success" />
  return <Wrench />
}

export function ToolHeader({
  name,
  status
}: {
  name: string
  status: ToolStatus
}) {
  return (
    <Disclosure.Heading>
      <Button
        slot="trigger"
        variant="ghost"
        size="sm"
        fullWidth
        className="justify-start rounded-3xl px-3 text-muted"
      >
        <StatusIcon status={status} />
        <span className="min-w-0 truncate font-mono text-xs text-foreground">
          {name}
        </span>
        <span className="shrink-0 text-xs">{STATUS_LABELS[status]}</span>
        <Disclosure.Indicator className="ms-auto" />
      </Button>
    </Disclosure.Heading>
  )
}

export function ToolContent({ children }: { children: ReactNode }) {
  return (
    <Disclosure.Content>
      <Disclosure.Body className="flex flex-col gap-2 border-t border-separator px-3 py-3 text-xs">
        {children}
      </Disclosure.Body>
    </Disclosure.Content>
  )
}

/** A value as readable text: JSON pretty-printed, text as it is. */
function asText(value: unknown) {
  if (typeof value !== "string") return JSON.stringify(value, null, 2)
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

/** The arguments the model wrote. */
export function ToolInput({ input }: { input: unknown }) {
  return (
    <>
      <p className="text-muted">Arguments</p>
      <pre className="max-h-48 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap">
        {asText(input ?? {})}
      </pre>
    </>
  )
}

/** What the tool returned, or why it failed. Nothing until either. */
export function ToolOutput({
  output,
  errorText
}: {
  output?: unknown
  errorText?: string
}) {
  if (output === undefined && !errorText) return null
  return (
    <>
      <p className="text-muted">{errorText ? "Error" : "Result"}</p>
      <pre
        className={`max-h-60 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap ${errorText ? "text-danger" : ""}`}
      >
        {errorText ?? asText(output)}
      </pre>
    </>
  )
}

/**
 * Asks before a tool runs: its name and arguments, with Approve and Deny.
 * Once answered, only the answer's status shows.
 */
export function ToolApproval({
  name,
  input,
  status,
  isDisabled,
  onApprove,
  onDeny
}: {
  name: string
  input: unknown
  status: "pending" | "validating" | "staged" | "submitting" | "error"
  isDisabled: boolean
  onApprove: () => void
  onDeny: () => void
}) {
  if (status !== "pending") {
    return <p className="text-xs text-muted">{status}</p>
  }
  return (
    <Card variant="secondary" className="gap-3 rounded-2xl p-4">
      <Card.Header className="gap-1 p-0">
        <Card.Title className="flex items-center gap-2 text-sm">
          <Hand className="text-warning" />
          Run <span className="font-mono text-xs">{name}</span>?
        </Card.Title>
        <Card.Description className="text-xs">
          This tool asks before it runs.
        </Card.Description>
      </Card.Header>
      <Card.Content className="p-0 text-xs">
        <ToolInput input={input} />
      </Card.Content>
      <Card.Footer className="gap-2 p-0">
        <Button size="sm" isDisabled={isDisabled} onPress={onApprove}>
          Approve
        </Button>
        <Button
          size="sm"
          variant="tertiary"
          isDisabled={isDisabled}
          onPress={onDeny}
        >
          Deny
        </Button>
      </Card.Footer>
    </Card>
  )
}
