# @tanstack-ai-browser/apfel

A [TanStack AI](https://tanstack.com/ai) adapter for Apple's on-device Foundation Model, served on your Mac by [apfel](https://apfel.franzai.com)'s OpenAI-compatible server. The browser streams straight from localhost: start it with `apfel --serve --cors`.

```sh
npm install @tanstack-ai-browser/apfel @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import { apfel, isApfelAvailable } from "@tanstack-ai-browser/apfel"

if (await isApfelAvailable()) {
  const stream = chat({ adapter: apfel(), messages, threadId })
}
```

**Status:** early (0.0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
