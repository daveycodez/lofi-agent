# @tanstack-ai-browser/apfel

A [TanStack AI](https://tanstack.com/ai) adapter for Apple's on-device Foundation Model, served on your Mac by [apfel](https://apfel.franzai.com)'s OpenAI-compatible server. The browser streams straight from apfel's server, at the `baseURL` you give it.

```sh
npm install @tanstack-ai-browser/apfel @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import {
  APFEL_BASE_URL,
  apfelText,
  isApfelAvailable
} from "@tanstack-ai-browser/apfel"

const config = { baseURL: APFEL_BASE_URL }

if (await isApfelAvailable("apple-foundationmodel", config)) {
  const stream = chat({
    adapter: apfelText("apple-foundationmodel", config),
    messages,
    threadId
  })
}
```

`baseURL` is apfel's OpenAI-compatible API, up to `/v1`; `APFEL_BASE_URL` is apfel's default address. `defaultHeaders` adds headers to every request, such as a bearer token. Apple's model doesn't reason, so `chat({ reasoning })` is a type error with this adapter.

Each call's token counts are in `RUN_FINISHED`'s `usage`, as with any TanStack AI adapter, with timings the adapter measured in `usage.providerUsageDetails` (`ApfelProviderUsageDetails`: `firstTokenMs`, `decodeMs`, `decodeTokens`).

```ts
// For example, in chat() middleware:
const stats = { name: "stats", onUsage: (ctx, usage) => console.log(usage) }
```

**Status:** early (0.0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
