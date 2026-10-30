/**
 * Tool calls, built from HeroUI's Disclosure and sized like HeroUI Pro's
 * ChatTool: a 12px-rounded bordered box, a `text-xs` trigger with a 14px
 * status icon, and the arguments and result as small code blocks.
 */
import CircleCheck from "@gravity-ui/icons/CircleCheck"
import CircleDashed from "@gravity-ui/icons/CircleDashed"
import CircleExclamation from "@gravity-ui/icons/CircleExclamation"
import CircleXmark from "@gravity-ui/icons/CircleXmark"
import Hand from "@gravity-ui/icons/Hand"
import { Button, Disclosure, Spinner } from "@heroui/react"
import type { ReactNode } from "react"

import { highlightJson, highlightThemeCss } from "#/lib/highlight"

/** The bordered, collapsible box every tool piece sits in. */
function ToolBox({
  tone = "default",
  defaultExpanded,
  children
}: {
  tone?: "default" | "warning"
  defaultExpanded?: boolean
  children: ReactNode
}) {
  return (
    <Disclosure
      defaultExpanded={defaultExpanded}
      className={`overflow-hidden rounded-xl border ${
        tone === "warning" ? "border-warning/40" : "border-border"
      }`}
    >
      {children}
    </Disclosure>
  )
}

/** The trigger row: whatever leads it, then the chevron at the far end. */
function ToolTrigger({ children }: { children: ReactNode }) {
  return (
    <Disclosure.Heading>
      <Disclosure.Trigger className="flex w-full items-center gap-2 px-3 py-2 text-start text-xs">
        {children}
        <Disclosure.Indicator className="ms-auto size-3.5 shrink-0 text-muted" />
      </Disclosure.Trigger>
    </Disclosure.Heading>
  )
}

/** The open box's contents, in HeroUI's default body padding. */
function ToolBody({
  gap = "gap-1",
  children
}: {
  gap?: string
  children: ReactNode
}) {
  return (
    <Disclosure.Content>
      <Disclosure.Body>
        <div className={`flex flex-col p-1 ${gap}`}>{children}</div>
      </Disclosure.Body>
    </Disclosure.Content>
  )
}

/** Two or more tool calls in a row, under one "N tool calls" box. */
export function ToolGroup({
  count,
  children
}: {
  count: number
  children: ReactNode
}) {
  return (
    <ToolBox>
      <ToolTrigger>
        {count === 1 ? "1 tool call" : `${count} tool calls`}
      </ToolTrigger>
      <ToolBody gap="gap-1.5">{children}</ToolBody>
    </ToolBox>
  )
}

/** One tool call, collapsed under its status and name. */
export function Tool({ children }: { children: ReactNode }) {
  return <ToolBox>{children}</ToolBox>
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
  preparing: "Using tool:",
  running: "Using tool:",
  approval: "Waiting to use:",
  error: "Tool failed:",
  done: "Used tool:",
  stopped: "Didn't use:"
}

function StatusIcon({ status }: { status: ToolStatus }) {
  const icon = "size-3.5 shrink-0"
  if (status === "preparing" || status === "running") {
    return <Spinner size="sm" color="current" className={`${icon} text-muted`} />
  }
  if (status === "approval") return <Hand className={`${icon} text-warning`} />
  if (status === "error") return <CircleXmark className={`${icon} text-danger`} />
  if (status === "stopped") {
    return <CircleDashed className={`${icon} text-muted`} />
  }
  return <CircleCheck className={`${icon} text-muted`} />
}

/** "Used tool: name", led by its status icon. */
function ToolLabel({
  icon,
  label,
  name
}: {
  icon: ReactNode
  label: string
  name: string
}) {
  return (
    <>
      {icon}
      <span className="shrink-0 text-muted">{label}</span>
      <span className="min-w-0 truncate font-medium">{name}</span>
    </>
  )
}

export function ToolHeader({
  name,
  status
}: {
  name: string
  status: ToolStatus
}) {
  return (
    <ToolTrigger>
      <ToolLabel
        icon={<StatusIcon status={status} />}
        label={STATUS_LABELS[status]}
        name={name}
      />
    </ToolTrigger>
  )
}

export function ToolContent({ children }: { children: ReactNode }) {
  return <ToolBody>{children}</ToolBody>
}

/** A value parsed as JSON, if it is JSON (or a string holding JSON). */
function asJson(value: unknown): { json: unknown } | null {
  if (typeof value !== "string") return { json: value }
  try {
    return { json: JSON.parse(value) }
  } catch {
    return null
  }
}

/** Code in a small rounded block: JSON highlighted, anything else as text. */
function CodeBlock({
  value,
  pretty,
  isError = false
}: {
  value: unknown
  /** Indented JSON, rather than on one line. */
  pretty: boolean
  isError?: boolean
}) {
  const parsed = asJson(value)
  const code = parsed
    ? JSON.stringify(parsed.json, null, pretty ? 2 : undefined)
    : String(value)
  return (
    // `.markdown` scopes the highlight theme's token colors.
    <div className="markdown my-1 overflow-hidden rounded-xl bg-surface-secondary">
      <style href="markdown-highlight" precedence="default">
        {highlightThemeCss}
      </style>
      <pre
        className={`max-h-72 overflow-auto px-2.5 py-1.5 font-mono text-[11px] leading-5 whitespace-pre-wrap ${isError ? "text-danger" : ""}`}
      >
        {parsed && !isError
          ? highlightJson(code).map((token, i) =>
              token.className ? (
                <span key={i} className={`th-token th-${token.className}`}>
                  {token.value}
                </span>
              ) : (
                token.value
              )
            )
          : code}
      </pre>
    </div>
  )
}

/** The arguments the model wrote, on one line. */
export function ToolInput({ input }: { input: unknown }) {
  return <CodeBlock value={input ?? {}} pretty={false} />
}

/** What the tool returned, or why it failed. Nothing until either. */
export function ToolOutput({
  output,
  errorText
}: {
  output?: unknown
  errorText?: string
}) {
  if (errorText) return <CodeBlock value={errorText} pretty isError />
  if (output === undefined) return null
  return <CodeBlock value={output} pretty />
}

/**
 * Asks before a tool runs: a tool box with a warning border, open on the
 * arguments, with Reject and Approve. Once answered, only its status shows.
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
    <ToolBox tone="warning" defaultExpanded>
      <ToolTrigger>
        <ToolLabel
          icon={<CircleExclamation className="size-3.5 shrink-0 text-muted" />}
          label="Approval needed:"
          name={name}
        />
      </ToolTrigger>
      <ToolBody>
        <ToolInput input={input} />
        <div className="flex justify-end gap-2 pt-3">
          <Button
            size="sm"
            variant="secondary"
            isDisabled={isDisabled}
            onPress={onDeny}
          >
            Reject
          </Button>
          <Button
            size="sm"
            variant="primary"
            isDisabled={isDisabled}
            onPress={onApprove}
          >
            Approve
          </Button>
        </div>
      </ToolBody>
    </ToolBox>
  )
}
