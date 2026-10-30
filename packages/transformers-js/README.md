# @tanstack-ai-browser/transformers-js

A [TanStack AI](https://tanstack.com/ai) adapter for [Transformers.js](https://huggingface.co/docs/transformers.js) models that run in the browser, on WebGPU in a Web Worker, tuned for Safari on iPhone.

```sh
npm install @tanstack-ai-browser/transformers-js @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import { transformersJs } from "@tanstack-ai-browser/transformers-js"

const stream = chat({
  adapter: transformersJs("daveycodez/LFM3-1.7B-Web", {
    toolFormat: "lfm3",
    sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.1 }
  }),
  messages,
  threadId,
  // Any level but "off" turns on the model's thinking.
  reasoning: "high"
})
```

`chat()` runs tools and the agent loop; each model call runs in the worker. Weights download once to the Origin Private File System, prompts are read in time-sized steps (under Safari's GPU watchdog), and the model's cache is reused across calls by token prefix, with the system prompt and tools saved to disk.

The model runs in a module Web Worker that the package starts with `new Worker(new URL("./worker.js", import.meta.url))`. Bundlers that understand that pattern (Vite, webpack 5, Rollup with a worker plugin) emit the worker file for you. With Vite, leave the package out of dependency pre-bundling, so the worker's URL still points at a real file:

```ts
// vite.config.ts
export default defineConfig({
  worker: { format: "es" },
  optimizeDeps: { exclude: ["@tanstack-ai-browser/transformers-js"] }
})
```

**Status:** early (0.0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
