# tanstack-ai-browser

[TanStack AI](https://tanstack.com/ai) adapters for language models that run in the browser. Hand one to `chat()` like any hosted model, and tools, sub-agents and `useChat` work the same, with the model on the user's device.

| Package                                                            | What it runs                                                                                                                       |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| [`@tanstack-ai-browser/transformers-js`](packages/transformers-js) | [Transformers.js](https://huggingface.co/docs/transformers.js) models on WebGPU in a Web Worker, tuned to work in Safari on iPhone |
| [`@tanstack-ai-browser/apfel`](packages/apfel)                     | Apple's on-device Foundation Model, served on a Mac by [apfel](https://apfel.franzai.com)                                          |

**Status:** early (0.0.x); the API may change. The [lofi-agent](apps/lofi-agent) demo app runs on them.

## Usage

```ts
import { chat } from "@tanstack/ai"
import { stream, useChat } from "@tanstack/ai-react"
import { transformersJs } from "@tanstack-ai-browser/transformers-js"

const adapter = transformersJs("daveycodez/LFM3-1.7B-Web", {
  toolFormat: "lfm3",
  sampling: { temperature: 0.1, top_k: 50, repetition_penalty: 1.1 }
})

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
packages/transformers-js/       @tanstack-ai-browser/transformers-js
packages/apfel/                 @tanstack-ai-browser/apfel
apps/lofi-agent/                the demo app, hosted on Cloudflare Workers
```

```sh
pnpm install
pnpm nx dev lofi-agent                 # the demo, http://localhost:3000
pnpm nx run-many -t typecheck lint     # every project
```

### Releasing

Releases run from GitHub Actions: **Actions → Release → Run workflow**. [Nx Release](https://nx.dev/features/manage-releases) works out each package's next version from its conventional commits since its last tag (or takes the version you type), updates its `CHANGELOG.md`, commits and tags it (`transformers-js@0.0.2`), publishes it to npm with provenance, and creates a GitHub release. Tick **dry run** to see what would happen first.

Publishing uses npm [trusted publishing](https://docs.npmjs.com/trusted-publishers), set up on npm for both packages with the workflow `release.yml`, so no npm token is stored in the repository.

You need Node.js and pnpm (the repo pins `pnpm@12.6.0` in `packageManager`). See the [demo's README](apps/lofi-agent/README.md) for its features, requirements and deployment.

## License

MIT
