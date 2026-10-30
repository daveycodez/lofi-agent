import Hand from "@gravity-ui/icons/Hand"
import TriangleExclamation from "@gravity-ui/icons/TriangleExclamation"
import Wrench from "@gravity-ui/icons/Wrench"
import { Button, Card, Disclosure, Spinner } from "@heroui/react"
import type { ReactNode } from "react"

/** A tool call, collapsed under its name. */
export function Tool({ children }: { children: ReactNode }) {
  return <Disclosure>{children}</Disclosure>
}

/** `approval`: the call waits for the user to approve or deny it. */
export type ToolStatus = "running" | "approval" | "error" | "done"

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
        size="sm"
        variant="ghost"
        className="-ms-3 text-muted"
      >
        {status === "running" ? (
          <Spinner size="sm" color="current" />
        ) : status === "approval" ? (
          <Hand className="text-warning" />
        ) : status === "error" ? (
          <TriangleExclamation className="text-danger" />
        ) : (
          <Wrench />
        )}
        <span className="font-mono text-xs">{name}</span>
        {/* Buttons pull icons in by 2px; keep the caret 8px from the label. */}
        <Disclosure.Indicator className="ms-0" />
      </Button>
    </Disclosure.Heading>
  )
}

export function ToolContent({ children }: { children: ReactNode }) {
  return (
    <Disclosure.Content>
      <Disclosure.Body className="flex flex-col gap-2 border-s-2 border-separator ps-3 text-xs">
        {children}
      </Disclosure.Body>
    </Disclosure.Content>
  )
}

/** Pretty-prints JSON, or returns the text as it is. */
function prettyJson(json: string) {
  try {
    return JSON.stringify(JSON.parse(json), null, 2)
  } catch {
    return json
  }
}

/** The arguments the model wrote, as JSON. */
export function ToolInput({ input }: { input: string }) {
  return (
    <>
      <p className="text-muted">Arguments</p>
      <pre className="max-h-48 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap">
        {prettyJson(input)}
      </pre>
    </>
  )
}

/** What the tool returned, or why it failed. Nothing until either. */
export function ToolOutput({
  output,
  errorText
}: {
  output?: string
  errorText?: string
}) {
  if (output === undefined && !errorText) return null
  return (
    <>
      <p className="text-muted">{errorText ? "Error" : "Result"}</p>
      <pre
        className={`max-h-60 overflow-auto rounded-xl bg-surface-secondary p-3 font-mono whitespace-pre-wrap ${errorText ? "text-danger" : ""}`}
      >
        {errorText ?? output}
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
  /** The arguments, as JSON. */
  input: string
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
