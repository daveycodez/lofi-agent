/**
 * Sources under an answer, and the inline citation chips that point to them.
 * Styled after HeroUI Pro's chat sources (a "N sources" toggle above favicon
 * chips), built from HeroUI OSS parts.
 */
import Globe from "@gravity-ui/icons/Globe"
import { Disclosure, Link, Tooltip } from "@heroui/react"
import { useState } from "react"

import type { Source } from "#/lib/sources"

/**
 * Google's favicon service, where google.com/s2/favicons redirects. Asked
 * directly: the redirect has no Cross-Origin-Resource-Policy header, so
 * the page's cross-origin isolation would block it.
 */
const faviconUrl = (url: string) =>
  `https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=${encodeURIComponent(url)}&size=64`

/** The site's icon, or a globe if it has none. */
function Favicon({ url }: { url: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) return <Globe aria-hidden className="size-3.5 shrink-0" />
  return (
    <img
      alt=""
      src={faviconUrl(url)}
      loading="lazy"
      // No CORS: Google's icons send no CORS headers, but allow embedding
      // in cross-origin isolated pages (Cross-Origin-Resource-Policy).
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      className="size-3.5 shrink-0 rounded-full"
    />
  )
}

/**
 * A source as a 20px pill like HeroUI Pro's ChatSource: 14px icon and the
 * muted site name, with a preview card on hover.
 */
function SourceChip({ source, label }: { source: Source; label?: string }) {
  return (
    <Tooltip delay={300}>
      <Link
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex h-5 max-w-32 items-center gap-1 overflow-hidden rounded-full bg-default px-1.5 py-0 text-xs leading-none font-normal text-muted no-underline transition-colors duration-150 hover:bg-default-hover hover:text-foreground"
      >
        <Favicon url={source.url} />
        <span className="min-w-0 truncate text-start">
          {label ?? source.site}
        </span>
      </Link>
      {/* The site, the page's title, and an excerpt when there is one. */}
      <Tooltip.Content className="w-80 rounded-2xl p-4">
        <span className="flex flex-col gap-1.5 p-1">
          <span className="flex min-w-0 items-center gap-1.5">
            <Favicon url={source.url} />
            <span className="truncate text-sm text-foreground">
              {source.site}
            </span>
          </span>
          <span className="line-clamp-2 text-sm font-medium">
            {source.title}
          </span>
          {source.snippet && (
            <span className="line-clamp-2 text-sm text-muted">
              {source.snippet}
            </span>
          )}
        </span>
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
      <Disclosure.Content className="border-default data-[expanded=true]:border-t">
        <Disclosure.Body>
          <div className="flex flex-wrap items-center gap-2 pt-2">
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
