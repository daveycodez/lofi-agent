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

## Tool calls

Tools work as with any adapter: pass them to `chat({ tools })`, MCP tools included, and `chat()` runs them. Apple's model picks the right tool but often gets its arguments wrong, renaming fields (`limit` for `numResults`) or dropping optional ones it was asked to set. So the adapter repairs every tool call before `chat()` runs it, using only the tool's JSON Schema:

1. For each optional field, apfel's guided generation answers yes or no: does the request give it a value?
2. The arguments are written again, constrained to the schema, with those fields required. Fields the model already named and typed right are kept as it wrote them.

Schema keywords the on-device model can't enforce (`minLength`, `pattern`, `format`…) are dropped for the repair, `$ref`s are inlined, and `anyOf`/`oneOf` take their first non-null choice. If a repair request fails, the model's own arguments are used. Repair adds two short requests per tool call (one for a tool without optional fields), about a second and a half in all; it's in `usage.providerUsageDetails.toolRepairMs`.

## Usage

Each call's token counts are in `RUN_FINISHED`'s `usage`, as with any TanStack AI adapter, with timings the adapter measured in `usage.providerUsageDetails` (`ApfelProviderUsageDetails`: `firstTokenMs`, `decodeMs`, `decodeTokens`, `toolRepairMs`).

```ts
// For example, in chat() middleware:
const stats = { name: "stats", onUsage: (ctx, usage) => console.log(usage) }
```

**Status:** early (0.0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
