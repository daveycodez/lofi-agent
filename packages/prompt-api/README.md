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

- `promptApiAvailability(languages?, { samplingMode?, inputs? })` says whether the browser can run its model: `"available"` (on the device), `"downloadable"` or `"downloading"` (it downloads when a session starts), or `"unavailable"` (no Prompt API, or the device can't run it). Give it what your sessions will use, as Chrome asks: `inputs: ["image"]` or `["audio"]` if users send them (audio needs a GPU).
- `promptApiStatus()` is a store of the model's loading, in the same shape as the other adapters' (`idle`, `loading` with `progress` 0 to 1 and `cached`, `ready`, `error`). Chrome reports a fraction only, never bytes. The first session `chat()` starts reports into it, so nothing needs loading first; its download of Gemini Nano took about 85 seconds on a fast connection, and loading it into memory takes about 6 to 8 seconds the first time after Chrome starts.
- `loadPromptApiModel()` downloads and loads the model ahead of the first message. Chrome wants a click on the page before it downloads (when availability is `"downloadable"`), so call it from the click that turns the model on.
- `promptApiModel()` says which model this browser provides: `{ id: "gemini-nano", name: "Gemini Nano", browser: "Chrome" }`, `{ id: "phi-4-mini", name: "Phi-4-mini", browser: "Edge" }`, or `"unknown"` for other browsers. The API itself doesn't say; the browser's brand does.
- `promptApiContextWindow()` gives the tokens a session holds (9,216 for Gemini Nano in Chrome 156).

`languages` (`["en"]` by default) tells the model which languages to expect in and out; Chrome's model takes `en`, `es`, `ja`, `de` and `fr`, and checks its safety filters for the output language. The model doesn't reason, so `chat({ reasoning })` is a type error with this adapter. `modelOptions.samplingMode` picks a sampling preset (`"most-predictable"` to `"most-creative"`); Chrome takes it on web pages only in its Sampling Parameters origin trial. `temperature` and `topK` work in extensions only.

## Images and audio

User messages can carry images and audio (TanStack AI's `image` and `audio` parts, as base64 data or a URL); they're loaded into Blobs only when a session reads them, and the session is created expecting them. A URL from another site needs CORS. The model can't write images or audio, and it can't read video or documents.

## Tool calls

Pass tools to `chat({ tools })` as with any adapter, MCP tools included; `chat()` runs them.

The Prompt API spec has tool use: the session writes tool calls, the page runs them and sends the results back. Where the browser has it (its `LanguageModelToolCall` classes, and a session with tools available), the adapter uses it. Chrome 156 doesn't have it on web pages yet. `promptApiText({ nativeTools: false })` turns it off.

Without it, tools go through guided generation (`responseConstraint` holds a reply to a JSON Schema), in two short constrained calls on copies of the session (`clone()`) before anything else:

1. The decision: `{"action": "answer"}` or the name of a tool.
2. For a tool, its arguments, held to the tool's input schema (in the subset guided generation enforces: types, properties, required, items, enum, const, descriptions).

An answer then streams from the session unconstrained, with Markdown. Asked to write tool calls as text instead, Gemini Nano made up an answer about current events rather than search; with the decision constrained it picked right in 10 of 10 test prompts. A step costs about 0.3 seconds for the decision, plus 0.2 to 2 seconds to write a tool's arguments. The model sees past calls as `{"tool", "arguments"}` JSON and results as `Result of <tool>:` messages. `toolChoice` works: `"none"` skips the decision, `"required"` leaves answering out, and a named tool skips straight to its arguments (both always use guided generation).

`chat({ outputSchema })` uses guided generation too, with the schema as given and, if the browser can't enforce it, in the subset above.

## Sessions

Each thread keeps one session, which holds the conversation the model has read. When the next request extends it (a follow-up, or the step after a tool call), only the new messages are read; anything else (an edited message, another system prompt, tools or options, an image in a text-only thread) starts a new session, and the adapter's logger says why. Only the system prompt is an initial prompt; the conversation is appended after it, so when a thread outgrows the context the browser drops its oldest turns (never the system prompt), and a new session leaves out the oldest turns that don't fit. When the browser drops turns, `RUN_FINISHED`'s `usage.providerUsageDetails.contextOverflow` is `true` and `promptApiText({ onContextOverflow })` is called. A single message too long for the model fails with an error that says so.

An aborted or failed call frees its session, and the next request starts again from the messages. Aborting a request never destroys the session a thread keeps, and a kept session the browser destroyed is replaced on the next request. `destroyPromptApiSessions()` frees every kept session.

## Usage

Each call's token counts are in `RUN_FINISHED`'s `usage`: `promptTokens` (the session's whole context), `completionTokens`, and `promptTokensDetails.cachedTokens` for what a reused session had already read. The reply is measured once it's done, so counting never holds up the first token; the counts are close, not exact (the Prompt API counts whole messages). Timings are in `usage.providerUsageDetails` (`PromptApiProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`, `contextUsage`, `contextWindow`, `contextOverflow`). With Gemini Nano in Chrome on an M3 Ultra: about 55 to 60 tokens/s, the first token 70 ms after a short prompt (0.3 s with tools, for the decision), and prompts read at 700 to 1,700 tokens/s.

**Status:** early (0.0.x); the API may change, and so does the Prompt API. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
