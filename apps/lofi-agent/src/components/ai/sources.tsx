/**
 * Sources under an answer, and the inline citation chips that point to them.
 * Styled after HeroUI Pro's chat sources (a "N sources" toggle above favicon
 * chips), built from HeroUI OSS parts.
 */
import BookOpen from "@gravity-ui/icons/BookOpen"
import Globe from "@gravity-ui/icons/Globe"
import { Disclosure, Link, Tooltip } from "@heroui/react"
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

/** A source as a pill: icon and site name, with the page title on hover. */
function SourceChip({ source, label }: { source: Source; label?: string }) {
  return (
    <Tooltip delay={300}>
      <Link
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex h-7 max-w-48 items-center gap-1.5 rounded-full bg-default px-2.5 align-middle text-xs text-muted no-underline hover:text-foreground"
      >
        <Favicon url={source.url} />
        <span className="truncate">{label ?? source.site}</span>
      </Link>
      <Tooltip.Content className="max-w-72">{source.title}</Tooltip.Content>
    </Tooltip>
  )
}

/** The sources an answer drew on, under a "Used N sources" toggle. */
export function SourceList({ sources }: { sources: Array<Source> }) {
  return (
    // 12px from the answer: the reply's 8px gap plus mt-1.
    <Disclosure className="mt-1">
      <Disclosure.Heading>
        {/* The divider under the toggle shows only while the list is open. */}
        <Disclosure.Trigger className="flex w-full items-center gap-2 border-separator pb-2 text-sm text-muted aria-expanded:border-b">
          <BookOpen aria-hidden className="size-4" />
          {sources.length === 1 ? "Used 1 source" : `Used ${sources.length} sources`}
          {/* Right after the label, not pushed to the far end. */}
          <Disclosure.Indicator className="ms-0" />
        </Disclosure.Trigger>
      </Disclosure.Heading>
      {/* No body padding: the chips sit 12px under the divider, flush left. */}
      <Disclosure.Content className="[&>.disclosure\_\_body]:p-0">
        <Disclosure.Body className="flex flex-wrap gap-2 pt-3">
          {sources.map((source) => (
            <SourceChip key={source.url} source={source} />
          ))}
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
