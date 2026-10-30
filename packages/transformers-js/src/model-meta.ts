/**
 * The models this package knows: Hugging Face ONNX builds that run in the
 * browser, with what each supports, the sampling from its card, and how it
 * writes tool calls. Any other Hugging Face id works too (see
 * `transformersJsText`), with the on/off reasoning toggle.
 */
import type { ModelReasoning } from "@tanstack/ai"
import type { ModelReasoningCapability } from "@tanstack/ai/adapter-internals"

import type { Sampling, WeightsDtype } from "./protocol.js"
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
  /** The weights to load: "q4f16" unless the model needs fp32 activations. */
  dtype?: WeightsDtype
  /**
   * The repo also has q4 weights (float32 activations), so the model runs
   * on the CPU through WebAssembly where there's no WebGPU.
   */
  cpu?: boolean
  /** Sampling from the model card, used unless `modelOptions` overrides it. */
  sampling: Sampling
}

// Reasoning shapes, each level's value being the chat template's
// `enable_thinking`. Read from the models' chat templates: LFM3's and
// Gemma 4's take `enable_thinking`; LFM2.5 Thinking always thinks and takes
// no switch.
const THINKING_SWITCH = {
  levels: { off: "false", high: "true" },
  budget: false
} as const
const ALWAYS_THINKS = { levels: { high: "true" }, budget: false } as const

/**
 * The LFM builds hold 8,192 tokens per chat in this runtime: past 4,096
 * tokens ONNX Runtime's buffer cache moves LFM3's attention memory to 8 MiB
 * buffers, which hold exactly 8,192; one more token would move it to
 * 12 MiB, past what iPhone Safari allows.
 */
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

/** LFM2 1.2B Tool card: greedy decoding (temperature 0). */
const LFM2_TOOL_SAMPLING = {
  temperature: 0,
  top_k: 1,
  repetition_penalty: 1
} as const satisfies Sampling

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

/** Gemma 4's generation config: temperature 1, top-k 64, top-p 0.95. */
const GEMMA_4_SAMPLING = {
  temperature: 1,
  top_k: 64,
  top_p: 0.95,
  repetition_penalty: 1
} as const satisfies Sampling

/**
 * Gemma 4 E4B (4.5B effective parameters, 8B with its per-layer
 * embeddings), in onnx-community's Transformers.js build; its text model
 * only (embedding and decoder, 4.9 GB).
 */
const GEMMA_4_E4B = {
  name: "onnx-community/gemma-4-E4B-it-ONNX",
  revision: "2b6d2e26a962f70a33da8f7cd81e1afdd8d9cfb2",
  supports: TOOLS_AND_THINKING,
  size: "4b",
  context: 32_768,
  reasoning: THINKING_SWITCH,
  toolFormat: "gemma4",
  sampling: GEMMA_4_SAMPLING
} as const satisfies TransformersJsModelMeta

/**
 * LFM3 built for iPhone Safari: GPTQ 4-bit with block size 32 (so
 * onnxruntime-web's fast prompt kernel and fused kernels apply), a 48k
 * vocabulary, no embedding copy at load (900 MB of weights instead of
 * 1.09 GB), and flash attention in the sliding-window layers.
 */
const LFM3_1_7B_WEB = {
  name: "daveycodez/LFM3-1.7B-Web",
  revision: "9cda8fa27706d239b9c58f53a867d398e313b961",
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  toolFormat: "lfm3",
  sampling: LFM3_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

/** The onnx-community export of LFM3. */
const LFM3_1_7B = {
  name: "onnx-community/LFM3-1.7B-ONNX",
  supports: TOOLS_AND_THINKING,
  size: "1.7b",
  context: CONTEXT,
  reasoning: THINKING_SWITCH,
  toolFormat: "lfm3",
  sampling: LFM3_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B = {
  name: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B_THINKING = {
  name: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
  supports: TOOLS_AND_THINKING,
  size: "1.2b",
  context: CONTEXT,
  reasoning: ALWAYS_THINKS,
  toolFormat: "lfm2",
  sampling: LFM2_5_THINKING_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

/**
 * Liquid AI's LFM2 1.2B Tool, in onnx-community's build: tuned for short,
 * precise tool calls without thinking. Its q4 weights (1.2 GB) run on the
 * CPU too.
 */
const LFM2_1_2B_TOOL = {
  name: "onnx-community/LFM2-1.2B-Tool-ONNX",
  revision: "6a719165fdb804a2e312e14bf1491221717a3049",
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_TOOL_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

/**
 * Liquid AI's LFM2.5 350M (instruction-tuned): the smallest and quickest,
 * for short answers, extraction and tool calls rather than knowledge or
 * code. Its graph already computes logits for the last token only.
 */
const LFM2_5_350M = {
  name: "LiquidAI/LFM2.5-350M-ONNX",
  revision: "d11593fd9eb408e322667926656598896c2d5ff9",
  supports: TOOLS,
  size: "350m",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

/**
 * LiquidAI's LFM2.5 re-exported with a `num_logits_to_keep` input, so
 * reading a prompt computes logits for its last token only (an 800-token
 * prompt otherwise needs 210 MB for them, which crashed iPhone Safari).
 */
const LFM2_5_1_2B_WEB = {
  name: "daveycodez/LFM2.5-1.2B-Web",
  revision: "f8df717df1ee01ccb899484d7f25cddfa78dd49f",
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

const LFM2_5_1_2B_THINKING_WEB = {
  name: "daveycodez/LFM2.5-1.2B-Thinking-Web",
  revision: "db6d0161174b69db3bd4ed789e9a1cb888344f32",
  supports: TOOLS_AND_THINKING,
  size: "1.2b",
  context: CONTEXT,
  reasoning: ALWAYS_THINKS,
  toolFormat: "lfm2",
  sampling: LFM2_5_THINKING_SAMPLING,
  cpu: true
} as const satisfies TransformersJsModelMeta

export const TRANSFORMERS_JS_TEXT_MODELS = [
  GEMMA_4_E4B.name,
  LFM3_1_7B_WEB.name,
  LFM3_1_7B.name,
  LFM2_5_1_2B_WEB.name,
  LFM2_5_1_2B_THINKING_WEB.name,
  LFM2_5_1_2B.name,
  LFM2_5_1_2B_THINKING.name,
  LFM2_1_2B_TOOL.name,
  LFM2_5_350M.name
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
  [GEMMA_4_E4B.name]: GEMMA_4_E4B,
  [LFM3_1_7B_WEB.name]: LFM3_1_7B_WEB,
  [LFM3_1_7B.name]: LFM3_1_7B,
  [LFM2_5_1_2B_WEB.name]: LFM2_5_1_2B_WEB,
  [LFM2_5_1_2B_THINKING_WEB.name]: LFM2_5_1_2B_THINKING_WEB,
  [LFM2_5_1_2B.name]: LFM2_5_1_2B,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING,
  [LFM2_1_2B_TOOL.name]: LFM2_1_2B_TOOL,
  [LFM2_5_350M.name]: LFM2_5_350M
}

const TRANSFORMERS_JS_REASONING = {
  [GEMMA_4_E4B.name]: GEMMA_4_E4B.reasoning,
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
export const modelMeta = (
  model: string
): TransformersJsModelMeta | undefined =>
  LISTED.has(model)
    ? TRANSFORMERS_JS_MODEL_META[model as TransformersJsListedModel]
    : undefined

/**
 * Whether a model runs without WebGPU, on the CPU: a listed model that has
 * q4 weights. Another Hugging Face id may too; it isn't known here, so the
 * answer is yes, and loading says if its repo has no q4 weights.
 */
export const runsWithoutWebGpu = (model: string): boolean =>
  LISTED.has(model) ? (modelMeta(model)?.cpu ?? false) : true

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
