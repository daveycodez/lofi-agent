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
  // The dev server also answers on the local network (http://<this mac>.local:3001)
  // and through ngrok tunnels, which give it the HTTPS that WebGPU needs.
  server: {
    host: true,
    allowedHosts: [".local", ".ngrok.dev", ".ngrok.app", ".ngrok-free.app"]
  },
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
