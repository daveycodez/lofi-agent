# @tanstack-ai-browser/prompt-api

A [TanStack AI](https://tanstack.com/ai) adapter for the browser's built-in language model, through the [Prompt API](https://developer.chrome.com/docs/ai/prompt-api) (`LanguageModel`): Gemini Nano in Chrome, Phi-4-mini in Edge. The browser downloads the model once for every site and runs it on the device; the page downloads nothing and nothing leaves the device.

```sh
npm install @tanstack-ai-browser/prompt-api @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import {
  promptApiAvailability,
  promptApiText
} from "@tanstack-ai-browser/prompt-api"

if ((await promptApiAvailability()) !== "unavailable") {
  const stream = chat({ adapter: promptApiText(), messages, threadId })
}
```

The Prompt API is in desktop Chrome and Edge (tested in Chrome 156, where web pages have it without flags), on devices with enough memory, GPU and disk (Chrome asks for 22 GB free and either a GPU with more than 4 GB or 16 GB of RAM). It isn't in Web Workers, and a cross-origin iframe needs `allow="language-model"`.

## The model

- `promptApiAvailability()` says whether the browser can run its model: `"available"` (on the device), `"downloadable"` or `"downloading"` (it downloads when a session starts), or `"unavailable"` (no Prompt API, or the device can't run it).
- `loadPromptApiModel()` downloads the model if needed and loads it; `chat()` calls it on the first message anyway. `promptApiStatus()` is a store of its progress, in the same shape as the other adapters' (`idle`, `loading` with `progress` 0 to 1 and `cached`, `ready`, `error`). Chrome reports a fraction only, not bytes. Its download of Gemini Nano took about 85 seconds on a fast connection; loading it into memory takes about 6 seconds the first time after Chrome starts.
- `promptApiModel()` says which model this browser provides: `{ id: "gemini-nano", name: "Gemini Nano", browser: "Chrome" }`, `{ id: "phi-4-mini", name: "Phi-4-mini", browser: "Edge" }`, or `"unknown"` for other browsers. The API itself doesn't say; the browser's brand does.
- `promptApiContextWindow()` gives the tokens a session holds (9,216 for Gemini Nano in Chrome 156).

`languages` (`["en"]` by default) tells the model which languages to expect in and out; Chrome's model takes `en`, `es`, `ja`, `de` and `fr`, and checks its safety filters for the output language. The model doesn't reason, so `chat({ reasoning })` is a type error with this adapter. `modelOptions` takes `samplingMode`, and `temperature` and `topK` for extensions; Chrome ignores them on web pages outside its Sampling Parameters origin trial.

## Tool calls

The Prompt API has no tool use for web pages yet, but it has guided generation: `responseConstraint` holds a reply to a JSON Schema. So tools work as with any adapter (pass them to `chat({ tools })`, MCP tools included, and `chat()` runs them), in two short constrained calls on copies of the session (`clone()`) before anything else:

1. The decision: `{"action": "answer"}` or the name of a tool.
2. For a tool, its arguments, held to the tool's input schema (in the subset guided generation enforces: types, properties, required, items, enum, const, descriptions).

An answer then streams from the session unconstrained, with Markdown. Asked to write tool calls as text instead, Gemini Nano made up an answer about current events rather than search; with the decision constrained it picked right in 10 of 10 test prompts. A step costs about 0.3 seconds for the decision, plus 0.4 to 2 seconds to write a tool's arguments. The model sees past calls as `{"tool", "arguments"}` JSON and results as `Result of <tool>:` messages. `toolChoice` works: `"none"` skips the decision, `"required"` leaves answering out, and a named tool skips straight to its arguments.

`chat({ outputSchema })` uses guided generation too, with the schema as given and, if the browser can't enforce it, in the subset above.

## Sessions

Each thread keeps one session, which holds the conversation the model has read. When the next request extends it (a follow-up, or the step after a tool call), only the new messages are read; anything else (an edited message, another system prompt or tools) starts a new session from the whole conversation. A conversation that outgrows the context loses its oldest turns, as the browser does. A single message too long for the model fails with an error that says so. An aborted or failed call frees its session, and the next request starts again from the messages.

## Usage

Each call's token counts are in `RUN_FINISHED`'s `usage`: `promptTokens` (the session's whole context), `completionTokens`, and `promptTokensDetails.cachedTokens` for what a reused session had already read. Timings are in `usage.providerUsageDetails` (`PromptApiProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`, `contextUsage`, `contextWindow`). With Gemini Nano in Chrome on an M3 Ultra: about 55 to 60 tokens/s, the first token 70 ms after a short prompt (0.3 s with tools, for the decision), and prompts read at 700 to 1,700 tokens/s.

**Status:** early (0.0.x); the API may change, and so does the Prompt API. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
