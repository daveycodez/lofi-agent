# @tanstack-ai-browser/transformers-js

A [TanStack AI](https://tanstack.com/ai) adapter for [Transformers.js](https://huggingface.co/docs/transformers.js) models that run in the browser, on WebGPU in a Web Worker, tuned for Safari on iPhone. Without WebGPU, models with CPU weights run on the CPU instead.

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

| Model                                 | Reasoning                  | Without WebGPU |
| ------------------------------------- | -------------------------- | -------------- |
| `onnx-community/gemma-4-E4B-it-ONNX`  | on/off                     | no             |
| `daveycodez/LFM3-1.7B-Web`            | on/off (`"off"`, `"high"`) | yes            |
| `onnx-community/LFM3-1.7B-ONNX`       | on/off                     | yes            |
| `daveycodez/LFM2.5-1.2B-Web`          | none                       | yes            |
| `daveycodez/LFM2.5-1.2B-Thinking-Web` | always (`"high"`)          | yes            |
| `LiquidAI/LFM2.5-1.2B-Instruct-ONNX`  | none                       | yes            |
| `LiquidAI/LFM2.5-1.2B-Thinking-ONNX`  | always                     | yes            |
| `onnx-community/LFM2-1.2B-Tool-ONNX`  | none                       | yes            |

Any other Hugging Face ONNX model works too, given how it writes tool calls and thinks: `transformersJsText("owner/model", { toolFormat: "gemma4" })`, with `"lfm2"`, `"lfm3"` or `"gemma4"`. Pin a revision with `"owner/model@commit"`. Its `q4f16` weights load by default; `dtype: "q4"` loads 4-bit weights with fp32 activations instead. `contextLength` sets how many tokens a chat may hold (8,192 by default). Weights split into several files, and builds split into an embedding, a decoder and vision or audio encoders, load as they are.

**Reasoning** works as with other TanStack AI adapters: `chat({ reasoning })` takes the model's levels, read from its chat template (`enable_thinking`), and is a type error for a model that doesn't reason. A model the package doesn't list gets the on/off toggle. `supportedReasoningLevels(TRANSFORMERS_JS_MODEL_REASONING[model])` from `@tanstack/ai` gives a listed model's levels at runtime.

**Sampling** goes in `modelOptions` per call (`temperature`, `top_k`, `repetition_penalty`, `max_new_tokens`), over the model card's.

**Usage** is in each call's `RUN_FINISHED` event, as with any TanStack AI adapter: `promptTokens`, `completionTokens`, `totalTokens`, and `promptTokensDetails.cachedTokens` for the start of the prompt the model reused from an earlier call. Timings are in `usage.providerUsageDetails` (`TransformersJsProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`), for example in `chat()` middleware's `onUsage`.

**Without WebGPU** (Firefox on Android, for one), there's nothing to set: the worker sees there's no GPU adapter and runs the model on the CPU through ONNX Runtime's WebAssembly backend, from its `q4` weights. It's slower, with the same features, prompt caching in memory and on disk included. `hasWebGpu()` says which this browser gets, and `runsWithoutWebGpu(model)` whether a model has CPU weights. It uses several threads where the page is cross-origin isolated, else one.

**Downloads:** `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its loading progress (with bytes downloaded and the total while it downloads).

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

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
