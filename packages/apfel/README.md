# @tanstack-ai-browser/apfel

A [TanStack AI](https://tanstack.com/ai) adapter for Apple's on-device Foundation Model, served on your Mac by [apfel](https://apfel.franzai.com)'s OpenAI-compatible server. The browser streams straight from localhost: start it with `apfel --serve --cors`.

```ts
import { chat } from "@tanstack/ai"
import { apfel, isApfelAvailable } from "@tanstack-ai-browser/apfel"

if (await isApfelAvailable()) {
  const stream = chat({ adapter: apfel(), messages, threadId })
}
```

**Status:** in development, not published yet. See the [repository README](../../README.md).
