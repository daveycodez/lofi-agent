import { createFileRoute } from "@tanstack/react-router"

/**
 * One session, by id. The chat itself is rendered by the layout around this
 * page (src/routes/_chat.tsx), which reads the id from the URL.
 */
export const Route = createFileRoute("/_chat/sessions/$sessionId")({})
