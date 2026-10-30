/**
 * TanStack AI adapter for the browser's built-in language model, through
 * the Prompt API: Gemini Nano in Chrome, Phi-4-mini in Edge.
 */
export { PromptApiTextAdapter, promptApiText } from "./adapter.js"
export type {
  PromptApiModelOptions,
  PromptApiProviderUsageDetails,
  PromptApiTextConfig,
  PromptApiTextModel
} from "./adapter.js"
export {
  loadPromptApiModel,
  promptApiAvailability,
  promptApiContextWindow,
  promptApiModel,
  promptApiStatus
} from "./model.js"
export type { Availability, ModelStatus, PromptApiModelInfo } from "./model.js"
export { destroyPromptApiSessions } from "./sessions.js"
export type { Store } from "./store.js"
