import { createFileRoute, redirect } from "@tanstack/react-router"

/**
 * A new session. The chat itself is rendered by the layout around this page
 * (src/routes/_chat.tsx).
 */
export const Route = createFileRoute("/_chat/")({
  // Sessions used to open at `/?thread=<id>`: send those links to the
  // session's own URL.
  validateSearch: (search: Record<string, unknown>): { thread?: string } => ({
    thread: typeof search.thread === "string" ? search.thread : undefined
  }),
  beforeLoad: ({ search }) => {
    if (search.thread) {
      throw redirect({
        to: "/sessions/$sessionId",
        params: { sessionId: search.thread },
        replace: true
      })
    }
  }
})
