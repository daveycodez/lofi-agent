/**
 * TanStack AI adapter for Apple's on-device Foundation Model, served by apfel.
 */
export {
  APFEL_BASE_URL,
  APFEL_TEXT_MODELS,
  ApfelTextAdapter,
  apfelText,
  isApfelAvailable
} from "./adapter.js"
export type {
  ApfelModelOptions,
  ApfelProviderUsageDetails,
  ApfelTextConfig,
  ApfelTextModel
} from "./adapter.js"
