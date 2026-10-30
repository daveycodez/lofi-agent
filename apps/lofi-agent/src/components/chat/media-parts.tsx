/**
 * Parts the local models don't write but a tool or another model could:
 * images, audio, video, documents, an MCP App's UI, and a tool result with
 * no call to show it in. Laid out as in TanStack's example.
 */
import FileText from "@gravity-ui/icons/FileText"
import { Chip, Link } from "@heroui/react"
import type { PartProps } from "@tanstack/ai-react/ui"

import type { chatOptions } from "#/chat/options"

/**
 * A content source as something the browser can load. A provider file
 * handle is an opaque id, so there is nothing to show for it.
 */
function sourceHref(source: {
  type: string
  value: string
  mimeType?: string
}): string | undefined {
  if (source.type === "file") return undefined
  if (source.type === "data") {
    return `data:${source.mimeType ?? "application/octet-stream"};base64,${source.value}`
  }
  return source.value
}

function UnrenderableSource({ label }: { label: string }) {
  return <p className="text-xs text-muted">{label} stored as a provider file</p>
}

export function ImagePart({ part }: PartProps<typeof chatOptions, "image">) {
  const src = sourceHref(part.source)
  if (!src) return <UnrenderableSource label="Image" />
  return (
    <img
      alt=""
      src={src}
      className="max-h-[341px] max-w-full self-start rounded-2xl object-cover"
    />
  )
}

export function AudioPart({ part }: PartProps<typeof chatOptions, "audio">) {
  const src = sourceHref(part.source)
  if (!src) return <UnrenderableSource label="Audio" />
  return <audio controls src={src} className="w-full" />
}

export function VideoPart({ part }: PartProps<typeof chatOptions, "video">) {
  const src = sourceHref(part.source)
  if (!src) return <UnrenderableSource label="Video" />
  return (
    <video controls src={src} className="max-h-[341px] max-w-full self-start rounded-2xl" />
  )
}

export function DocumentPart({
  part
}: PartProps<typeof chatOptions, "document">) {
  const href = sourceHref(part.source)
  if (!href) return <UnrenderableSource label="Document" />
  const mime = part.source.type === "data" ? part.source.mimeType : undefined
  return (
    // A pill like the source chips: file icon, then what it is.
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="w-fit no-underline"
    >
      <Chip size="sm" className="h-5 gap-1 px-1.5 text-muted hover:text-foreground">
        <FileText className="size-3.5 shrink-0" />
        <Chip.Label className="px-0 text-xs leading-3">
          {mime ? `Document (${mime})` : "Document"}
        </Chip.Label>
      </Chip>
    </Link>
  )
}

/** A tool result whose call isn't in this message (a call's own shows in its card). */
export function ToolResultPart({
  part
}: PartProps<typeof chatOptions, "toolResult">) {
  return (
    <p className="text-xs text-muted">
      {typeof part.content === "string" ? part.content : "Tool result"}
    </p>
  )
}

/** An MCP App's UI, from its tool, in a sandboxed frame. */
export function UIResourcePart({
  part
}: PartProps<typeof chatOptions, "uiResource">) {
  const html = part.resource.text
  return (
    // The same bordered box as a tool call.
    <div className="flex flex-col gap-2 rounded-xl border border-border p-3">
      <p className="text-xs text-muted">
        <span className="font-medium text-foreground">{part.toolName}</span>
      </p>
      <div>
        {html ? (
          <iframe
            title={part.resource.uri}
            sandbox=""
            srcDoc={html}
            className="h-48 w-full rounded-lg bg-background"
          />
        ) : (
          <p className="text-xs text-muted">{part.resource.uri}</p>
        )}
      </div>
    </div>
  )
}
