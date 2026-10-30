# @tanstack-ai-browser/transformers-js

A [TanStack AI](https://tanstack.com/ai) adapter for [Transformers.js](https://huggingface.co/docs/transformers.js) models that run in the browser, on WebGPU in a Web Worker, tuned for Safari on iPhone.

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

**Status:** in development, not published yet. See the [repository README](../../README.md).
