# @tanstack-ai-browser/transformers-js

A [TanStack AI](https://tanstack.com/ai) adapter for [Transformers.js](https://huggingface.co/docs/transformers.js) models that run in the browser, on WebGPU in a Web Worker, tuned for Safari on iPhone.

```sh
npm install @tanstack-ai-browser/transformers-js @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import { transformersJsText } from "@tanstack-ai-browser/transformers-js"

const stream = chat({
  adapter: transformersJsText("daveycodez/LFM3-1.7B-Web"),
  messages,
  threadId,
  reasoning: "high"
})
```

## Models

`TRANSFORMERS_JS_TEXT_MODELS` lists the models this package knows, and `TRANSFORMERS_JS_MODEL_META` has each one's details: Hugging Face id and pinned revision, size, context, capabilities, tool format, reasoning levels and the sampling from its model card. A listed model needs no config.

| Model                                 | Reasoning                  |
| ------------------------------------- | -------------------------- |
| `daveycodez/LFM3-1.7B-Web`            | on/off (`"off"`, `"high"`) |
| `onnx-community/LFM3-1.7B-ONNX`       | on/off                     |
| `daveycodez/LFM2.5-1.2B-Web`          | none                       |
| `daveycodez/LFM2.5-1.2B-Thinking-Web` | always (`"high"`)          |
| `LiquidAI/LFM2.5-1.2B-Instruct-ONNX`  | none                       |
| `LiquidAI/LFM2.5-1.2B-Thinking-ONNX`  | always                     |

Any other Hugging Face ONNX model works too, given how it writes tool calls: `transformersJsText("owner/model", { toolFormat: "lfm3" })`. Pin a revision with `"owner/model@commit"`.

**Reasoning** works as with other TanStack AI adapters: `chat({ reasoning })` takes the model's levels, read from its chat template (`enable_thinking`), and is a type error for a model that doesn't reason. A model the package doesn't list gets the on/off toggle. `supportedReasoningLevels(TRANSFORMERS_JS_MODEL_REASONING[model])` from `@tanstack/ai` gives a listed model's levels at runtime.

**Sampling** goes in `modelOptions` per call (`temperature`, `top_k`, `repetition_penalty`, `max_new_tokens`), over the model card's.

**Usage** is in each call's `RUN_FINISHED` event, as with any TanStack AI adapter: `promptTokens`, `completionTokens`, `totalTokens`, and `promptTokensDetails.cachedTokens` for the start of the prompt the model reused from an earlier call. Timings are in `usage.providerUsageDetails` (`TransformersJsProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`), for example in `chat()` middleware's `onUsage`.

**Downloads:** `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its loading progress.

## How it runs

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
