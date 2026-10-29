import { defineConfig } from "vite"
import { devtools } from "@tanstack/devtools-vite"

import { tanstackStart } from "@tanstack/react-start/plugin/vite"

import viteReact from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { cloudflare } from "@cloudflare/vite-plugin"

const config = defineConfig({
  resolve: { tsconfigPaths: true },
  // Module workers, so the LFM3 worker can import Transformers.js from the CDN.
  worker: { format: "es" },
  plugins: [
    devtools(),
    cloudflare({ viteEnvironment: { name: "ssr" } }),
    tailwindcss(),
    tanstackStart({
      // Prerender every page reachable from "/" to static HTML at build time.
      prerender: { enabled: true, crawlLinks: true }
    }),
    viteReact()
  ]
})

export default config
