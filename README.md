# lofi-agent

A chat app that runs small language models entirely in your browser. Models are downloaded from Hugging Face once, stored on your device, and run on the GPU through WebGPU in a Web Worker, so your prompts and replies are never sent to a server. It is built to work in Safari on iPhone, not just on desktop, and can also talk to Apple's on-device Foundation Model on a Mac. The models can call tools: a built-in web search and any MCP servers you add.

## Why it's different

- **Private.** In-browser models run on your device. The app has no backend and no accounts: the Cloudflare Worker only serves prerendered pages. Sessions, settings and MCP server details stay in your browser's storage.
- **Runs on iPhone.** The default model is a custom build made for iPhone Safari, and the runtime is tuned to stay inside its memory and GPU limits. In-browser models hold up to 8,192 tokens per session, prompt and reply together.
- **Download once.** Model weights (about 0.8 to 1.1 GB per model) are saved to the browser's Origin Private File System and loaded from disk on later visits.
- **Reuses work between runs.** The model keeps what it has already read, so a follow-up message, the next step after a tool call, or a new session with the same system prompt only reads what's new. The system prompt and tools are also saved to disk, so they aren't read again after a reload.

What does leave your device: web search queries (if web search is on), calls to MCP servers you add, and the downloads of model files (Hugging Face) and the runtime (jsDelivr).

## Models

| Model                       | Notes                                                                                                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LFM3 1.7B Web** (default) | Liquid AI's LFM3, in a custom ONNX build for iPhone Safari: 4-bit weights (about 900 MB), a smaller vocabulary, and flash attention in its sliding-window layers to keep memory down on long prompts. |
| LFM3 1.7B                   | The `onnx-community` ONNX export of LFM3.                                                                                                                                                             |
| LFM2.5 1.2B                 | Liquid AI's ONNX builds. With Think on, it runs the separate LFM2.5 1.2B Thinking model.                                                                                                              |
| LFM2.5 1.2B Web             | The same weights, re-exported so reading a prompt computes output for the last token only. Long prompts (for example, with web search results) crashed iPhone Safari without this.                    |
| Apple Foundation Model      | Apple's on-device model, served on your Mac by [apfel](https://apfel.franzai.com). Shown on Apple devices only; needs a Mac (see Requirements).                                                       |

The "Web" builds are published under [`daveycodez` on Hugging Face](https://huggingface.co/daveycodez) and pinned to a commit. Models that can reason before answering have a Think toggle in the composer, on by default.

## Features

- Streaming replies with collapsible thinking, Markdown, math (KaTeX) and syntax-highlighted code
- Web search tool, on by default: tries Exa, then Parallel, then Firecrawl through their public MCP endpoints. No API keys needed. Results show as numbered source chips the reply can cite.
- MCP servers: add any Streamable HTTP server by URL, with an optional bearer token. The browser connects directly, so the server must allow this site (CORS).
- Sessions saved in the browser (messages in IndexedDB, the session list in localStorage)
- Sidebar with search, pinned sessions and date groups; rename, pin, archive and delete sessions, or delete them all
- Edit a sent message and resend it, even while the model is still replying
- Per reply: copy, regenerate, read aloud, export as Markdown, delete
- Copy a whole session as plain text, including thinking, tool calls and their results
- A stats line under the composer for the latest reply: tokens in, cached, out and total, prompt reading and generation speed, and time to first token
- Voice input through the browser's speech recognition, where available
- Settings: turn web search on or off, and write a system prompt sent with every session (before the current date)
- Light, dark and system themes
- Add to Home Screen: installs as an app with its own icon, and keeps the downloaded model more reliably than a Safari tab

## Requirements

- A browser with WebGPU. Where the browser has JavaScript Promise Integration (Safari 27 and recent Chrome), a smaller, faster ONNX Runtime build is used.
- Around 1 GB of free storage per model.
- WebGPU needs a secure context: `localhost` during development, HTTPS otherwise.
- For the Apple Foundation Model: a Mac with Apple silicon on macOS 26 or later, with Apple Intelligence turned on, and apfel running:

  ```sh
  brew install apfel
  apfel --serve --cors
  ```

  On a site other than `localhost`, add `--allowed-origins <site origin>`. The app shows these steps when it can't reach the model. It isn't available on iPhone or iPad.

## Getting started

You need Node.js and pnpm (the repo pins `pnpm@12.6.0` in `packageManager`).

```sh
pnpm install
pnpm nx dev web
```

The app runs at http://localhost:3000. The dev server also listens on your local network and accepts ngrok tunnel hosts, which give it the HTTPS that WebGPU needs when you open it on a phone.

Other tasks:

```sh
pnpm nx build web       # production build, pages prerendered
pnpm nx preview web     # serve the build
pnpm nx typecheck web
pnpm nx lint web
pnpm nx deploy web      # build, then wrangler deploy to Cloudflare Workers
```

Deploying needs a Cloudflare account (`wrangler login`). No environment variables or API keys are required.

## Project structure

```
apps/web/                  the app (Nx project "web")
  src/routes/              pages (TanStack Router file routes)
  src/components/chat/     thread, composer, sidebar, model picker, Apple setup
  src/components/mcp/      MCP servers dialog
  src/components/settings/ settings dialog
  src/lib/llm/             models, the WebGPU worker, Apple model connection,
                           tool-call parsing, reply stats
  src/lib/                 web search, MCP, chat storage, settings, dictation,
                           Markdown helpers
  wrangler.jsonc           Cloudflare Worker config
```

## Tech stack

- [TanStack Start](https://tanstack.com/start) and TanStack Router with React 19, on Vite
- [TanStack AI](https://tanstack.com/ai) for chat state, tool calls, IndexedDB persistence and the MCP client
- [Transformers.js](https://huggingface.co/docs/transformers.js) 4.3.2 and onnxruntime-web on WebGPU, loaded from jsDelivr at runtime
- [HeroUI](https://heroui.com) v3, Tailwind CSS v4 and Gravity UI icons
- TanStack Markdown, tanstack-markdown-math (KaTeX) and TanStack Highlight
- Cloudflare Workers for hosting
- Nx and pnpm workspace

## Credits

- The chat and sidebar layout follows [assistant-ui](https://github.com/assistant-ui/assistant-ui) (MIT).
- Provider icons are from [LobeHub Icons](https://github.com/lobehub/lobe-icons) (MIT).
- LFM3 and LFM2.5 are models by [Liquid AI](https://huggingface.co/LiquidAI).
- Apple's model is reached through [apfel](https://apfel.franzai.com).
