/**
 * Every page the Worker renders is cross-origin isolated, so models on the
 * CPU (where there's no WebGPU) can run on several threads: WebAssembly
 * threads need SharedArrayBuffer, which only isolated pages get. COEP is
 * "require-corp", the one every browser honours (Safari and Firefox on
 * Android ignore "credentialless"): an image from another site loads only
 * with `crossOrigin="anonymous"` and CORS, or with a
 * Cross-Origin-Resource-Policy header; fetches and module scripts use CORS
 * anyway. Prerendered pages and static files get the same headers from
 * public/_headers, and the dev server from vite.config.ts.
 */
import { createMiddleware, createStart } from "@tanstack/react-start"
import { setResponseHeaders } from "@tanstack/react-start/server"

export const CROSS_ORIGIN_ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp"
} as const

const crossOriginIsolation = createMiddleware().server(({ next }) => {
  setResponseHeaders(new Headers(CROSS_ORIGIN_ISOLATION))
  return next()
})

export const startInstance = createStart(() => ({
  requestMiddleware: [crossOriginIsolation]
}))
