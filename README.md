# tanstack-ai-browser

[TanStack AI](https://tanstack.com/ai) adapters for language models that run in the browser. Hand one to `chat()` like any hosted model, and tools, sub-agents and `useChat` work the same, with the model on the user's device.

The first adapter runs [Transformers.js](https://huggingface.co/docs/transformers.js) models on WebGPU in a Web Worker, tuned to work in Safari on iPhone.

**Status:** in development, not published yet. The adapter is being built out of the [lofi-agent](apps/lofi-agent) demo app, which runs the same runtime today.

## Planned usage

```ts
import { chat } from "@tanstack/ai"
import { stream, useChat } from "@tanstack/ai-react"
import { transformersJs } from "@daveycodez/tanstack-ai-browser"

const adapter = transformersJs("daveycodez/LFM3-1.7B-Web")

// chat() runs in the page; useChat reads its stream directly.
const chatState = useChat({
  connection: stream((messages, _data, _signal, input) =>
    chat({ adapter, messages, threadId: input.threadId })
  )
})
```

Swapping the adapter switches models; swapping `useChat`'s connection to `fetchServerSentEvents` runs the same `chat()` on a server with a hosted model.

## What the runtime handles

- **Downloads once.** Weights are saved to the Origin Private File System and read back slice by slice, so a ~1 GB model fits in iPhone Safari's memory limit.
- **Reads long prompts safely.** Prompts are read in steps sized by time, to stay under Safari's 3-second GPU watchdog.
- **Reuses what it has read.** The model's cache is kept between runs and matched by token prefix, one per agent, and the system prompt and tools are saved to disk across reloads.
- **Works around WebGPU memory limits.** GPU buffer pooling and garbage-collector nudges keep long replies under the page's memory limit.

## Repository

```
packages/tanstack-ai-browser/   the library (@daveycodez/tanstack-ai-browser)
apps/lofi-agent/                the demo app, hosted on Cloudflare Workers
```

```sh
pnpm install
pnpm nx dev lofi-agent                 # the demo, http://localhost:3000
pnpm nx run-many -t typecheck lint     # every project
```

You need Node.js and pnpm (the repo pins `pnpm@12.6.0` in `packageManager`). See the [demo's README](apps/lofi-agent/README.md) for its features, requirements and deployment.

## License

MIT
