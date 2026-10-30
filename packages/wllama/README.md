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

`WLLAMA_TEXT_MODELS` lists the models this package knows, and `WLLAMA_MODEL_META` has each one's details: Hugging Face repository, pinned revision, GGUF file and its size, context, capabilities, tool format, reasoning levels and the sampling from its card. They're Liquid AI's official GGUF builds.

| Model                                | Reasoning         | File            |
| ------------------------------------ | ----------------- | --------------- |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | none              | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | always (`"high"`) | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-350M-GGUF`          | none              | Q4_K_M, 229 MB  |

Every model is Q4_K_M: in llama.cpp's WebAssembly build it reads prompts and writes tokens about three times as fast as Q4_0, and faster than Q5_K_M, Q6_K or Q8_0. On an M3 Ultra (24 threads), LFM2.5 1.2B reads 231 prompt tokens/s and writes 112 tokens/s; on one thread, 10 and 7.6. LFM3 isn't listed yet: the llama.cpp in wllama 3.8.1 predates its architecture.

**Tools** work as with any adapter: pass them to `chat({ tools })`, MCP tools included. LFM2.5 writes Pythonic calls between `<|tool_call_start|>` and `<|tool_call_end|>`; the adapter parses them into TanStack AI tool calls, and a reply ends at its calls.

**Reasoning:** LFM2.5 1.2B Thinking always thinks, and its thinking streams as reasoning events. `chat({ reasoning })` is a type error for the models that don't reason.

**Sampling** goes in `modelOptions` per call (`temperature`, `top_k`, `top_p`, `repeat_penalty`, `max_new_tokens`), over the model card's.

**Usage** is in each call's `RUN_FINISHED` event, as with any TanStack AI adapter: `promptTokens`, `completionTokens`, `totalTokens`, and `promptTokensDetails.cachedTokens` for the start of the prompt the model reused from the previous call. Timings are in `usage.providerUsageDetails` (`WllamaProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`).

**Downloads:** `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its loading progress (with bytes downloaded and the total while it downloads).

## How it runs

`chat()` runs tools and the agent loop; each model call is one completion in wllama's worker.

- **Downloads once.** The GGUF file is saved to the Origin Private File System by wllama, and loads from there afterwards, offline too.
- **Uses every core.** Where the page is cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin` with `Cross-Origin-Embedder-Policy: require-corp` or `credentialless`), llama.cpp runs on as many threads as the browser reports cores; elsewhere, one. `wllamaText(model, { threads })` sets a number.
- **Writes the prompt with the model's own chat template**, the Jinja template in its GGUF file, so tools and thinking appear as the template intends.
- **Reuses what it has read.** llama.cpp keeps what it read last, and each reply goes back into the next prompt exactly as the model wrote it, thinking and tool calls included, so a follow-up or the step after a tool call only reads what's new. LFM2's convolution state can't be cut back to a shorter prefix, so llama.cpp saves it before the last 128 tokens of each prompt: a new chat with the same system prompt and tools rereads at most those.

### Bundlers

The package starts wllama with `new URL("@wllama/wllama/esm/wasm/wllama.wasm", import.meta.url)`, which Vite resolves and emits as an asset, in its dev server and builds alike; nothing to configure. With a bundler that doesn't, serve that file yourself and pass its URL:

```ts
wllamaText("LiquidAI/LFM2.5-350M-GGUF", { wasmUrl: "/wllama.wasm" })
```

Browsers without JSPI or WebAssembly Memory64 (Safari) get wllama's compatibility build, which it fetches from jsDelivr (`@wllama/wllama-compat`). It's slower.

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
