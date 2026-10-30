# web

The lofi-agent chat app: TanStack Start on Cloudflare Workers, with models running in the browser on WebGPU. See the [repository README](../../README.md) for what it does, requirements and setup.

Run tasks from the repository root through Nx:

```sh
pnpm nx dev web         # http://localhost:3000
pnpm nx build web
pnpm nx preview web
pnpm nx typecheck web
pnpm nx lint web
pnpm nx deploy web      # build, then wrangler deploy
```

Where things are:

- `src/routes/` pages (TanStack Router file routes; `src/routeTree.gen.ts` is generated)
- `src/components/` chat thread, sidebar, model picker, MCP and settings dialogs
- `src/lib/llm/` model list, the WebGPU worker (`local-model.worker.ts`), the Apple model connection, tool-call parsing and reply stats
- `src/lib/` web search, MCP client, chat storage, settings, dictation
- `wrangler.jsonc` Cloudflare Worker config
