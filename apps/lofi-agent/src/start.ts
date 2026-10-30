/**
 * Every page the Worker renders is cross-origin isolated, so models on the
 * CPU (where there's no WebGPU) can run on several threads: WebAssembly
 * threads need SharedArrayBuffer, which only isolated pages get. COEP is
 * "credentialless", so images and scripts from other sites still load,
 * just without cookies. Prerendered pages and static files get the same
 * headers from public/_headers, and the dev server from vite.config.ts.
 */
import { createMiddleware, createStart } from "@tanstack/react-start"
import { setResponseHeaders } from "@tanstack/react-start/server"

export const CROSS_ORIGIN_ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "credentialless"
} as const

const crossOriginIsolation = createMiddleware().server(({ next }) => {
  setResponseHeaders(new Headers(CROSS_ORIGIN_ISOLATION))
  return next()
})

export const startInstance = createStart(() => ({
  requestMiddleware: [crossOriginIsolation]
}))
