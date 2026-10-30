/**
 * The models this package knows: GGUF files on Hugging Face that run on the
 * CPU with wllama, with what each supports, the sampling from its card, and
 * how it writes tool calls.
 */
import type { ModelReasoning } from "@tanstack/ai"
import type { ModelReasoningCapability } from "@tanstack/ai/adapter-internals"

import type { ToolFormat } from "./tool-calls.js"

/** Sampling settings from a model's card, as llama.cpp takes them. */
export type Sampling = {
  temperature: number
  top_k: number
  /** llama.cpp's repetition penalty (`--repeat-penalty`). */
  repeat_penalty: number
  /** Nucleus sampling: the likeliest tokens up to this probability. */
  top_p?: number
}

export type WllamaModelMeta = {
  /** The Hugging Face repository the GGUF file is in. */
  name: string
  /** The commit the file is pinned to, so browsers fetch that version. */
  revision: string
  /** The GGUF file in the repository. */
  file: string
  /** The file's size in bytes. */
  bytes: number
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

/** LFM2.5 Thinking always thinks; its chat template takes no switch. */
const ALWAYS_THINKS = { levels: { high: "true" }, budget: false } as const

/**
 * Tokens a chat may hold, as with the package's other LFM models: the
 * attention cache for 8,192 tokens is 96 MB, little next to the weights,
 * and a full context already takes half a minute to read on a phone.
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

/** The cards' llama.cpp settings: `--temp 0.1 --top-k 50 --repeat-penalty 1.05`. */
const LFM2_5_SAMPLING = {
  temperature: 0.1,
  top_k: 50,
  repeat_penalty: 1.05
} as const satisfies Sampling
/** LFM2.5 Thinking's card: `--temp 0.05 --top-k 50 --repeat-penalty 1.05`. */
const LFM2_5_THINKING_SAMPLING = {
  temperature: 0.05,
  top_k: 50,
  repeat_penalty: 1.05
} as const satisfies Sampling

// Every model is Q4_K_M: in wllama's WebAssembly build it reads prompts and
// writes tokens about three times as fast as Q4_0 (llama.cpp has no
// WebAssembly SIMD kernel for Q4_0 to match its Q4_K one), and faster than
// Q5_K_M, Q6_K and Q8_0, at 5% more size than Q4_0. LFM2.5 1.2B on an
// M3 Ultra, 24 threads: Q4_K_M 231 prompt tokens/s and 112 tokens/s,
// Q4_0 (Liquid's QAD build too) 65 and 45.
//
// LFM3 would go here, but the llama.cpp in wllama 3.8.1 (b11364) predates
// its architecture ("lfm3", added to llama.cpp on 2026-10-27).

/** Liquid AI's LFM2.5 1.2B Instruct, from its official GGUF repository. */
const LFM2_5_1_2B = {
  name: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
  revision: "8ed288026e23958ad9dfa92d53ed773a8eee7125",
  file: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
  bytes: 730_895_168,
  supports: TOOLS,
  size: "1.2b",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING
} as const satisfies WllamaModelMeta

/** LFM2.5 1.2B Thinking: thinks before every answer. */
const LFM2_5_1_2B_THINKING = {
  name: "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
  revision: "b71945805f7d4d87dabd73e26098d44bcad6c35f",
  file: "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
  bytes: 730_899_072,
  supports: TOOLS_AND_THINKING,
  size: "1.2b",
  context: CONTEXT,
  reasoning: ALWAYS_THINKS,
  toolFormat: "lfm2",
  sampling: LFM2_5_THINKING_SAMPLING
} as const satisfies WllamaModelMeta

/**
 * LFM2.5 350M: the smallest and quickest, for short answers, extraction and
 * tool calls rather than knowledge or code.
 */
const LFM2_5_350M = {
  name: "LiquidAI/LFM2.5-350M-GGUF",
  revision: "657e078c94084481950a2d555a941481f715536b",
  file: "LFM2.5-350M-Q4_K_M.gguf",
  bytes: 229_312_224,
  supports: TOOLS,
  size: "350m",
  context: CONTEXT,
  toolFormat: "lfm2",
  sampling: LFM2_5_SAMPLING
} as const satisfies WllamaModelMeta

export const WLLAMA_TEXT_MODELS = [
  LFM2_5_1_2B.name,
  LFM2_5_1_2B_THINKING.name,
  LFM2_5_350M.name
] as const

/** A model this package lists. */
export type WllamaTextModel = (typeof WLLAMA_TEXT_MODELS)[number]

/** Each listed model's metadata, by name. */
export const WLLAMA_MODEL_META: Readonly<
  Record<WllamaTextModel, WllamaModelMeta>
> = {
  [LFM2_5_1_2B.name]: LFM2_5_1_2B,
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING,
  [LFM2_5_350M.name]: LFM2_5_350M
}

const WLLAMA_REASONING = {
  [LFM2_5_1_2B_THINKING.name]: LFM2_5_1_2B_THINKING.reasoning
} as const

/** The reasoning levels of each listed model that reasons. */
export type WllamaModelReasoningByName = {
  [K in keyof typeof WLLAMA_REASONING]: ModelReasoningCapability<
    (typeof WLLAMA_REASONING)[K]
  >
}

/**
 * Reasoning data of each listed model that reasons, for
 * `supportedReasoningLevels()`. A model missing here doesn't reason.
 */
export const WLLAMA_MODEL_REASONING: Readonly<Record<string, ModelReasoning>> =
  WLLAMA_REASONING

/** A listed model's metadata. */
export const modelMeta = (model: WllamaTextModel): WllamaModelMeta =>
  WLLAMA_MODEL_META[model]

/** Where a model's GGUF file downloads from: its pinned commit on Hugging Face. */
export const modelUrl = (model: WllamaTextModel) => {
  const { name, revision, file } = modelMeta(model)
  return `https://huggingface.co/${name}/resolve/${revision}/${file}`
}
