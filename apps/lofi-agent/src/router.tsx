import { createRouter as createTanStackRouter } from "@tanstack/react-router"
import { routeTree } from "./routeTree.gen"

export function getRouter() {
  const router = createTanStackRouter({
    routeTree,
    // The app scrolls inside its own panels, never the window.
    scrollRestoration: false,
    defaultPreload: "intent",
    defaultPreloadStaleTime: 0
  })

  return router
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
