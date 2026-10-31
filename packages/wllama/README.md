# @tanstack-ai-browser/wllama

A [TanStack AI](https://tanstack.com/ai) adapter for GGUF models that run in the browser, with [wllama](https://github.com/ngxson/wllama): llama.cpp compiled to WebAssembly, in a Web Worker. It runs on WebGPU where the browser has it, and on the CPU elsewhere, so it runs in any browser with WebAssembly SIMD: Firefox on Android and iPhone Safari included.

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

## Setup

- **Vite and Next.js:** install and go. They emit the package's WebAssembly files, in development and in builds, with Turbopack or webpack.
- **No other dependency:** wllama's JavaScript ships in the package (`vendor/wllama/`), built with its WebAssembly files, rather than from `@wllama/wllama`, whose JavaScript can't load them.
- **Storage:** models download to the browser's Origin Private File System, and so does what each model has read of the system prompt and tools (10 to 22 MB a model), so the first message after a reload starts in about a second (see [How it runs](#how-it-runs)). Nothing to set up.
- **Threads (optional):** serve the site cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) and wllama runs on half the cores; otherwise on one. Set the headers on your host, and in development (below). With `require-corp`, cross-origin images and other resources load only with CORS (`<img crossOrigin="anonymous">`) or a `Cross-Origin-Resource-Policy` header.
- **Other bundlers:** serve the files in the package's `wasm/` and pass their URLs (see [Bundlers](#bundlers)).

```ts
// vite.config.ts
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp"
}
export default defineConfig({ server: { headers }, preview: { headers } })
```

```ts
// next.config.ts
const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" }
        ]
      }
    ]
  }
}
```

## Models

`WLLAMA_TEXT_MODELS` lists the models this package knows, and `WLLAMA_MODEL_META` has each one's details: Hugging Face repository, pinned revision, GGUF file and its size, context, capabilities, reasoning levels and the sampling from its card. They're Liquid AI's official GGUF builds, plus LFM3 Web, a leaner LFM3.

| Model                                | Reasoning                       | File            |
| ------------------------------------ | ------------------------------- | --------------- |
| `LiquidAI/LFM3-1.7B-GGUF`            | on or off (`"high"` or `"off"`) | Q4_K_M, 1.10 GB |
| `daveycodez/LFM3-1.7B-Web-GGUF`      | on or off (`"high"` or `"off"`) | Q4_K, 912 MB    |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | none                            | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | always (`"high"`)               | Q4_K_M, 731 MB  |
| `LiquidAI/LFM2.5-350M-GGUF`          | none                            | Q4_K_M, 229 MB  |

Liquid's models are Q4_K_M: on a Pixel 10 Pro in Firefox, LFM2.5 350M read prompts at 55 tokens/s in Q4_K_M, 40 in Q8_0 and 15 in Q4_0 with wllama's own build. With this package's faster build (below), LFM2.5 1.2B reads about 400 prompt tokens/s and writes 90 tokens/s in Chrome on an M3 Ultra (16 threads). LFM3 1.7B, with its system prompt and tools, reads about 180 prompt tokens/s and writes 70 tokens/s there.

LFM3 Web ([daveycodez/LFM3-1.7B-Web-GGUF](https://huggingface.co/daveycodez/LFM3-1.7B-Web-GGUF)) is LFM3 with the smaller vocabulary of the Transformers.js LFM3 Web build, in Q4_K with an importance matrix: 188 MB smaller than Liquid's Q4_K_M, it loads about 27% faster and writes about 6% faster, with the same quality in English and the same tool calls. Other languages take 1.6 to 4.2 times as many tokens, so prefer Liquid's build for them.

**Tools** work as with any adapter: pass them to `chat({ tools })`, MCP tools included. llama.cpp writes them into the prompt with the model's chat template and parses the model's calls back.

**Reasoning:** thinking (llama.cpp's `reasoning_content`) streams as reasoning events. LFM3 thinks by default; `chat({ reasoning: "off" })` turns it off, through its chat template's `enable_thinking` (llama.cpp's `chat_template_kwargs`). LFM2.5 1.2B Thinking always thinks. `chat({ reasoning })` is a type error for the models that don't reason.

**Sampling** goes in `modelOptions` per call (`temperature`, `top_k`, `top_p`, `repeat_penalty`, `max_new_tokens`), over the model card's. Without `max_new_tokens`, a reply may run to the context limit, as llama.cpp's default.

**Usage** is in each call's `RUN_FINISHED` event, as with any TanStack AI adapter: `promptTokens`, `completionTokens`, `totalTokens`, and `promptTokensDetails.cachedTokens` for the start of the prompt llama.cpp reused from an earlier call. Timings are in `usage.providerUsageDetails` (`WllamaProviderUsageDetails`: `firstTokenMs`, `promptMs`, `decodeMs`, `decodeTokens`).

**Downloads:** `loadModel(model)` downloads and loads a model ahead of time, and `modelStatus(model)` is a store of its loading progress (with bytes downloaded and the total while it downloads).

## How it runs

`chat()` runs tools and the agent loop; each model call is one chat completion from llama.cpp's OpenAI-compatible API (`createChatCompletion`), in wllama's worker. llama.cpp writes the prompt with the model's chat template, reads thinking and tool calls back, and reuses whatever start of the prompt it has already read. wllama's defaults apply, except:

- **Context:** each model's own (8,192 tokens), not wllama's 1,024, which a system prompt with tools outgrows.
- **WebGPU or CPU:** wllama's default: on WebGPU where the browser has it (Chrome, Edge, Safari's compat build), on the CPU elsewhere (Firefox, which has no JSPI, and browsers without WebGPU). In Chrome on an M3 Ultra, LFM3 1.7B Web read 575 prompt tokens/s and wrote 70 tokens/s on WebGPU, against 47 and 27 on the CPU (4 threads). `wllamaText(model, { gpu: false })` runs on the CPU only. This package's builds use llama.cpp's subgroup shaders only where the browser also has WGSL's `subgroup_id`; without that check, Chrome 141 couldn't compile them and the model didn't load.
- **Full sliding-window cache:** `swa_full: true`, so LFM3 reuses what it has read after a tool call (626 of 670 prompt tokens in our test) instead of rereading the whole prompt (0 reused by default). It costs 65 MB more for LFM3 at 8,192 tokens, and nothing for models without sliding-window layers.
- **Saved prompt across reloads:** after a chat, the model's state (what it has read, as llama.cpp's slot save file) is saved to the Origin Private File System, one per model, and saved again only when the system prompt or tools change. Loading the model restores it, so the first message after a reload reads only the new message: in Chrome on an M3 Ultra (4 threads), with a system prompt and five tools, LFM3 1.7B answered in 0.9 s instead of 15.6 s (698 of 722 prompt tokens reused), LFM2.5 1.2B in 0.5 s instead of 11.9 s (773 of 795). llama.cpp compares the restored tokens with the prompt as usual, so a state that doesn't match is just not reused; one from another build of wllama isn't restored. Saving takes about 0.1 s after a reply; the state is 22 MB for LFM3 and 11 MB for LFM2.5.
- **This package's builds of wllama:** wllama 3.8.1, built on a newer llama.cpp (`19cce69`, the commit that added LFM3) with llama.cpp's WebAssembly SIMD kernels, which wllama's own builds leave out (Emscripten reports the CPU as `x86`, so llama.cpp falls back to generic code), faster model loading (tensors read in file order, 64 MB read-ahead), slot state save and restore (`Wllama.saveSlotState()` and `restoreSlotState()`, for the saved prompt above), and context checkpoints at user messages, so LFM2.5, which can't roll its memory back, reuses a shared system prompt in a new chat (773 of 795 tokens instead of 274). wllama's JavaScript, in `vendor/wllama/`, is built from the same patched source, since the main builds' glue changes with them. Three builds ship in `wasm/`: `wllama-relaxed.wasm`, the main build with relaxed SIMD dot products as well, for Chrome and Firefox, which have WebAssembly relaxed SIMD; `wllama.wasm`, the main build without them, for other browsers; and `wllama-compat.wasm` with its JavaScript glue `wllama-compat.js.txt` (a `.txt` so bundlers emit it as is), the compat build, for Safari. Prompts read about twice as fast as with wllama's own build: on a Pixel 10 Pro in Firefox, LFM2.5 1.2B went from 16 to 32 tokens/s and its first token from 38 to 19 s. `scripts/build-wasm.sh` rebuilds them and `vendor/wllama/` (the patches it applies are next to it), and checks both main builds' glue is the one in that JavaScript; `wllamaText(model, { relaxedSimdWasmUrl: false })` uses the main build without relaxed SIMD.
- **Threads:** wllama's default, half the cores, where the page is cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin` with `Cross-Origin-Embedder-Policy: require-corp`; Safari and Firefox on Android ignore `credentialless`); one elsewhere. `wllamaText(model, { threads })` sets a number.

The GGUF file is saved to the Origin Private File System by wllama, and loads from there afterwards, offline too.

### Bundlers

The package starts wllama with `new URL("../wasm/wllama.wasm", import.meta.url)` and `new URL("../wasm/wllama-relaxed.wasm", import.meta.url)`, which Vite and Next.js (Turbopack and webpack) resolve and emit as assets, in development and builds alike; nothing to configure. (Vite's dev server pre-bundles the package, which moves those URLs where it serves no files, so the package then loads them from `/node_modules/@tanstack-ai-browser/wllama/wasm/`.) With a bundler that doesn't, serve those files from the package's `wasm/` yourself and pass their URLs:

```ts
wllamaText("LiquidAI/LFM2.5-350M-GGUF", {
  wasmUrl: "/wllama.wasm",
  relaxedSimdWasmUrl: "/wllama-relaxed.wasm"
})
```

`wasmUrl` alone loads that file in every browser that runs the main build.

Browsers that can't compile 64-bit WebAssembly memory or lack JSPI (Safari, on iPhone too) run the compat build instead, emitted the same way: `new URL("../wasm/wllama-compat.wasm", import.meta.url)` and `…/wllama-compat.js.txt`. It's slower. Elsewhere, serve both files and pass their URLs:

```ts
wllamaText("LiquidAI/LFM2.5-350M-GGUF", {
  compat: { wasm: "/wllama-compat.wasm", worker: "/wllama-compat.js.txt" }
})
```

**Status:** early (0.x); the API may change. See the [repository README](https://github.com/daveycodez/tanstack-ai-browser#readme).
