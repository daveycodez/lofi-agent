/**
 * The models this package knows: Hugging Face ONNX builds that run in the
 * browser, with what each supports, the sampling from its card, and how it
 * writes tool calls. Any other Hugging Face id works too (see
 * `transformersJsText`), with the on/off reasoning toggle.
 */
import type { ModelReasoning } from "@tanstack/ai"
import type { ModelReasoningCapability } from "@tanstack/ai/adapter-internals"

import type { Sampling } from "./protocol.js"
import type { ToolFormat } from "./tool-calls.js"

export type TransformersJsModelMeta = {
  /** The Hugging Face model id. */
  name: string
  /** The commit the model's files are pinned to, so browsers fetch that version. */
  revision?: string
  supports: {
    input: ReadonlyArray<"text">
    output: ReadonlyArray<"text">
    capabilities: ReadonlyArray<"tools" | "thinking">
  }
  /** Parameter count. */
  size: string
  /** Tokens a chat can hold, prompt and reply together, in this runtime. */
  context: number
  /** Reasoning levels for `chat({ reasoning })`, read from the chat template. */
  reasoning?: ModelReasoning
  /** How the model writes tool calls. */
  toolFormat: ToolFormat
  /** Sampling from the model card, used unless `modelOptions` overrides it. */
  sampling: Sampling
}

// Reasoning shapes, each level's value being the chat template's
// `enable_thinking`. Read from the models' chat templates: LFM3's takes
// `enable_thinking`; LFM2.5 Thinking always thinks and takes no switch.
const THINKING_SWITCH = {
  levels: { off: "false", high: "true" },
  budget: false
} as const
const ALWAYS_THINKS = { levels: { high: "true" }, budget: false } as const

/** The runtime holds 8,192 tokens per chat, the most iPhone Safari allows. */
const CONTEXT = 8192

const TOOLS_AND_THINKING = {
  input: ["text"],
  output: ["text"],
  capabilities: ["tools", "thinking"]
} as const
const TOOLS = {
  input: ["text"],
  output: ["text"],
  capabilities: ["tools"]
} as const

/** LFM3 card, "General" (chat, tool use, agents, coding). */
const LFM3_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.1
} as const satisfies Sampling
const LFM2_5_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repetition_penalty: 1.05
} as const satisfies Sampling
const LFM2_5_THINKING_SAMPLING = {
  temperature: 0.05,
  top_k: 50,
  repetition_penalty: 1.05
} as const satisfies Sampling

/**
 * LFM3 built for iPhone Safari: GPTQ 4-bit with block size 32 (so
 * onnxruntime-web's fast prompt kernel and fused kernels apply), a 48k
 * vocabulary, no embedding copy at load (900 MB of weights instead of
 * 1.09 GB), and flash attention in the sliding-window layers.
 */
const LFM3_1_7B_WEB = {
  name: "daveycodez/LFM3-1.7B-Web",
  revision: "3c75909fbf17af6f734263e9cee75625bcfd7523",
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  toolFormat: "lfm3",
  sampling: LFM3_SAMPLING
} as const satisfies TransformersJsModelMeta

/** The onnx-community export of LFM3. */
const LFM3_1_7B = {
  name: "onnx-community/LFM3-1.7B-ONNX",
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  toolFormat: "lfm3",
  sampling: LFM3_SAMPLING
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B = {
  name: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B_THINKING = {
  name: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
  supports: TOOLS_AND_THINKING,
  size: "1.2b",
  context: CONTEXT,
  reasoning: ALWAYS_THINKS,
  toolFormat: "lfm2",
  sampling: LFM2_5_THINKING_SAMPLING
} as const satisfies TransformersJsModelMeta

/**
 * LiquidAI's LFM2.5 re-exported with a `num_logits_to_keep` input, so
 * reading a prompt computes logits for its last token only (an 800-token
 * prompt otherwise needs 210 MB for them, which crashed iPhone Safari).
 */
const LFM2_5_1_2B_WEB = {
  name: "daveycodez/LFM2.5-1.2B-Web",
  revision: "7cbec27ef76aac9be538a16dd5dd57d19328c654",
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B_THINKING_WEB = {
  name: "daveycodez/LFM2.5-1.2B-Thinking-Web",
  revision: "eac258e512896325899cf2be2bbd18b546c71706",
  supports: TOOLS_AND_THINKING,
  size: "1.2b",
  context: CONTEXT,
  reasoning: ALWAYS_THINKS,
  toolFormat: "lfm2",
  sampling: LFM2_5_THINKING_SAMPLING
} as const satisfies TransformersJsModelMeta

export const TRANSFORMERS_JS_TEXT_MODELS = [
  LFM3_1_7B_WEB.name,
  LFM3_1_7B.name,
  LFM2_5_1_2B_WEB.name,
  LFM2_5_1_2B_THINKING_WEB.name,
  LFM2_5_1_2B.name,
  LFM2_5_1_2B_THINKING.name
] as const

/** A model this package lists. */
export type TransformersJsListedModel =
  (typeof TRANSFORMERS_JS_TEXT_MODELS)[number]

/** A listed model, or any other Hugging Face id. */
export type TransformersJsTextModel = TransformersJsListedModel | (string & {})

/** Each listed model's metadata, by name. */
export const TRANSFORMERS_JS_MODEL_META: Readonly<
  Record<TransformersJsListedModel, TransformersJsModelMeta>
> = {
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB,
  [LFM3_1_7B.name]: LFM3_1_7B,
  [LFM2_5_1_2B_WEB.name]: LFM2_5_1_2B_WEB,
  [LFM2_5_1_2B_THINKING_WEB.name]: LFM2_5_1_2B_THINKING_WEB,
  [LFM2_5_1_2B.name]: LFM2_5_1_2B,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING
}

const TRANSFORMERS_JS_REASONING = {
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB.reasoning,
  [LFM3_1_7B.name]: LFM3_1_7B.reasoning,
  [LFM2_5_1_2B_THINKING_WEB.name]: LFM2_5_1_2B_THINKING_WEB.reasoning,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING.reasoning
} as const

/** The reasoning levels of each listed model that reasons. */
export type TransformersJsModelReasoningByName = {
  [K in keyof typeof TRANSFORMERS_JS_REASONING]: ModelReasoningCapability<
    (typeof TRANSFORMERS_JS_REASONING)[K]
  >
}

/**
 * Reasoning data of each listed model that reasons, for
 * `supportedReasoningLevels()`. A listed model missing here doesn't reason.
 */
export const TRANSFORMERS_JS_MODEL_REASONING: Readonly<
  Record<string, ModelReasoning>
> = TRANSFORMERS_JS_REASONING

/** The on/off toggle for a model this package doesn't list. */
export const UNLISTED_MODEL_REASONING: ModelReasoning = THINKING_SWITCH

const LISTED: ReadonlySet<string> = new Set(TRANSFORMERS_JS_TEXT_MODELS)

/** A listed model's metadata, or undefined for any other id. */
export const modelMeta = (model: string): TransformersJsModelMeta | undefined =>
  LISTED.has(model)
    ? TRANSFORMERS_JS_MODEL_META[model as TransformersJsListedModel]
    : undefined

/** A model's reasoning data: listed, none for a listed non-reasoner, else the toggle. */
export const modelReasoning = (model: string): ModelReasoning | undefined =>
  TRANSFORMERS_JS_MODEL_REASONING[model] ??
  (LISTED.has(model) ? undefined : UNLISTED_MODEL_REASONING)

/**
 * The id the runtime loads: a listed model at its pinned revision
 * ("owner/repo@revision"), any other id as given.
 */
export function runtimeModelId(model: string) {
  const revision = modelMeta(model)?.revision
  return revision ? `${model}@${revision}` : model
}
