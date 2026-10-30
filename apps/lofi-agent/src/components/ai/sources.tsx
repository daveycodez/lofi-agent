/**
 * Sources under an answer, and the inline citation chips that point to them.
 * Styled after HeroUI Pro's chat sources (a "N sources" toggle above favicon
 * chips), built from HeroUI OSS parts.
 */
import Globe from "@gravity-ui/icons/Globe"
import { Chip, Disclosure, Link, Tooltip } from "@heroui/react"
import { useState } from "react"

import type { Source } from "#/lib/sources"

const faviconUrl = (url: string) =>
  `https://www.google.com/s2/favicons?domain_url=${encodeURIComponent(url)}&sz=64`

/** The site's icon, or a globe if it has none. */
function Favicon({ url }: { url: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) return <Globe aria-hidden className="size-3.5 shrink-0" />
  return (
    <img
      alt=""
      src={faviconUrl(url)}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      className="size-3.5 shrink-0 rounded-full"
    />
  )
}

/** A source as a small chip: icon and site name, with a preview on hover. */
function SourceChip({ source, label }: { source: Source; label?: string }) {
  return (
    <Tooltip delay={300}>
      <Link
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="no-underline"
      >
        {/* A 20px pill like HeroUI Pro's ChatSource: 14px icon, muted title. */}
        <Chip
          size="sm"
          className="h-5 max-w-48 gap-1 px-1.5 text-muted hover:text-foreground"
        >
          <Favicon url={source.url} />
          <Chip.Label className="truncate px-0 text-xs leading-3">
            {label ?? source.site}
          </Chip.Label>
        </Chip>
      </Link>
      {/* The site, the page's title, and an excerpt when there is one. */}
      <Tooltip.Content className="flex max-w-80 flex-col gap-2 rounded-3xl p-4">
        <span className="flex min-w-0 items-center gap-2 text-sm">
          <Favicon url={source.url} />
          <span className="truncate">{source.site}</span>
        </span>
        <span className="text-sm font-medium">{source.title}</span>
        {source.snippet && (
          <span className="line-clamp-3 text-sm text-muted">
            {source.snippet}
          </span>
        )}
      </Tooltip.Content>
    </Tooltip>
  )
}

/** The sources an answer drew on, under a "N sources" toggle. */
export function SourceList({ sources }: { sources: Array<Source> }) {
  return (
    <Disclosure>
      <Disclosure.Heading>
        {/* Sized like HeroUI Pro's ChatSources trigger, only as wide as its
            label and chevron. */}
        <Disclosure.Trigger className="flex min-h-8 w-fit items-center gap-2 py-1 text-xs font-medium text-muted transition-colors hover:text-foreground">
          {sources.length === 1 ? "1 source" : `${sources.length} sources`}
          <Disclosure.Indicator className="ms-0 size-3.5 shrink-0 text-muted" />
        </Disclosure.Trigger>
      </Disclosure.Heading>
      {/* A divider, then the chips, in HeroUI's default body padding. */}
      <Disclosure.Content className="border-border data-[expanded=true]:border-t">
        <Disclosure.Body>
          <div className="flex flex-wrap gap-2">
            {sources.map((source) => (
              <SourceChip key={source.url} source={source} />
            ))}
          </div>
        </Disclosure.Body>
      </Disclosure.Content>
    </Disclosure>
  )
}

/** A `[n]` citation in the answer, shown as a small source chip. */
export function CitationChip({ source }: { source: Source }) {
  return (
    <span className="not-prose mx-0.5 inline-flex">
      <SourceChip source={source} />
    </span>
  )
}
