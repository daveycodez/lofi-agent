# @tanstack-ai-browser/wllama

A [TanStack AI](https://tanstack.com/ai) adapter for GGUF models that run in the browser on the CPU, with [wllama](https://github.com/ngxson/wllama): llama.cpp compiled to WebAssembly, in a Web Worker. It needs no WebGPU, so it runs in any browser with WebAssembly SIMD: Firefox on Android and iPhone Safari included.

```sh
npm install @tanstack-ai-browser/wllama @tanstack/ai
```

```ts
import { chat } from "@tanstack/ai"
import { wllamaText } from "@tanstack-ai-browser/wllama"

const stream = chat({
  adapter: wllamaText("LiquidAI/LFM2.5-1.2B-Instruct-GGUF"),
  messages,
  threadId
})
```

## Models

`WLLAMA_TEXT_MODELS` lists the models this package knows, and `WLLAMA_MODEL_META` has each one's details: Hugging Face repository, pinned revision, GGUF file and its size, context, capabilities, reasoning levels and the sampling from its card. They're Liquid AI's official GGUF builds.

| Model                                | Reasoning         | File            |
| ------------------------------------ | ----------------- | --------------- |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | none              | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | always (`"high"`) | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-350M-GGUF`          | none              | Q4_K_M, 229 MB  |

Every model is Q4_K_M: in llama.cpp's WebAssembly build it reads prompts and writes tokens about three times as fast as Q4_0, and faster than Q5_K_M, Q6_K or Q8_0. On an M3 Ultra (24 threads), LFM2.5 1.2B reads 231 prompt tokens/s and writes 112 tokens/s; on one thread, 10 and 7.6. LFM3 isn't listed yet: the llama.cpp in wllama 3.8.1 predates its architecture.

**Tools** work as with any adapter: pass them to `chat({ tools })`, MCP tools included. llama.cpp writes them into the prompt with the model's chat template and parses the model's calls back.

**Reasoning:** LFM2.5 1.2B Thinking always thinks, and its thinking (llama.cpp's `reasoning_content`) streams as reasoning events. `chat({ reasoning })` is a type error for the models that don't reason.

**Sampling** goes in `modelOptions` per call (`temperature`, `top_k`, `top_p`, `repeat_penalty`, `max_new_tokens`), over the model card's. Without `max_new_tokens`, a reply may run to the context limit, as llama.cpp's default.

**Usage** is in each call's `RUN_FINISHED` event, as with any TanStack AI adapter: `promptTokens`, `completionTokens`, `totalTokens`, and `promptTokensDetails.cachedTokens` for the start of the prompt llama.cpp reused from an earlier call. Timings are in `usage.providerUsageDetails` (`WllamaProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`).

**Downloads:** `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its loading progress (with bytes downloaded and the total while it downloads).

## How it runs

`chat()` runs tools and the agent loop; each model call is one chat completion from llama.cpp's OpenAI-compatible API (`createChatCompletion`), in wllama's worker. llama.cpp writes the prompt with the model's chat template, reads thinking and tool calls back, and reuses whatever start of the prompt it has already read. wllama's defaults apply, except:

- **Context:** each model's own (8,192 tokens), not wllama's 1,024, which a system prompt with tools outgrows.
- **On the CPU:** `n_gpu_layers: 0`, so it runs the same everywhere, without WebGPU. wllama would otherwise use WebGPU where the browser has it.
- **Threads:** wllama's default, half the cores, where the page is cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin` with `Cross-Origin-Embedder-Policy: require-corp`; Safari and Firefox on Android ignore `credentialless`); one elsewhere. `wllamaText(model, { threads })` sets a number.

The GGUF file is saved to the Origin Private File System by wllama, and loads from there afterwards, offline too.

### Bundlers

The package starts wllama with `new URL("@wllama/wllama/esm/wasm/wllama.wasm", import.meta.url)`, which Vite resolves and emits as an asset, in its dev server and builds alike; nothing to configure. With a bundler that doesn't, serve that file yourself and pass its URL:

```ts
wllamaText("LiquidAI/LFM2.5-350M-GGUF", { wasmUrl: "/wllama.wasm" })
```

Browsers that can't compile 64-bit WebAssembly memory or lack JSPI (Safari, on iPhone too) run wllama's compat build instead, from `@wllama/wllama-compat`, emitted the same way: `new URL("@wllama/wllama-compat/wasm/wllama.wasm", import.meta.url)` and `…/wllama.js`. It's slower, on one thread. Elsewhere, serve both files and pass their URLs:

```ts
wllamaText("LiquidAI/LFM2.5-350M-GGUF", {
  compat: { wasm: "/wllama-compat.wasm", worker: "/wllama-compat.js" }
})
```

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
