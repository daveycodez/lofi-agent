/**
 * Tool calls for the built-in model. The Prompt API spec has tool use (open
 * loop: the session writes tool calls, the page runs them and sends back
 * their results), but Chrome doesn't ship it to web pages yet, so where the
 * browser lacks it (`supportsNativeTools`), tools work through guided
 * generation instead: `responseConstraint` holds the reply to a JSON Schema.
 * Each step that has tools works in two constrained calls, on copies of the
 * session (`clone()`), before anything is generated for real:
 *
 * 1. The decision: `{"action": "answer" | <tool name>}`.
 * 2. For a tool, its arguments, held to the tool's input schema.
 *
 * An answer then streams from the session as usual, unconstrained. Asked to
 * write tool calls as text instead, Gemini Nano made up answers to a
 * question about current events; with the decision constrained it picked
 * right 10 times in 10 (4 tool calls, 6 answers).
 */
import type { Tool } from "@tanstack/ai"

import { baseOptions } from "./model.js"
import { languageModel, toolClasses } from "./types.js"
import type { LanguageModelTool } from "./types.js"

type JsonSchema = Record<string, unknown>

/** The decision's reply, read. */
export type Decision = { action: string }

/** Keywords kept for guided generation; the others are dropped. */
const KEPT = ["type", "description", "enum", "const"] as const

const isObject = (value: unknown): value is JsonSchema =>
  !!value && typeof value === "object" && !Array.isArray(value)

/** The schema a local `$ref` (`#/$defs/X`, `#/definitions/X`) points to. */
function resolveRef(ref: string, root: JsonSchema): JsonSchema | undefined {
  if (!ref.startsWith("#/")) return undefined
  let node: unknown = root
  for (const key of ref.slice(2).split("/")) {
    node = isObject(node)
      ? node[key.replace(/~1/g, "/").replace(/~0/g, "~")]
      : undefined
  }
  return isObject(node) ? node : undefined
}

/**
 * A tool's input schema in the subset guided generation holds the model to:
 * types, properties, required, items, enum, const and descriptions, with
 * no other properties allowed. Local `$ref`s are inlined, `anyOf`/`oneOf`
 * and type unions become their first non-null choice, and constraints like
 * lengths, patterns and formats are dropped (Chrome rejects some of them,
 * and `chat()` validates the arguments anyway).
 */
export function toGuidedSchema(
  schema: unknown,
  root: JsonSchema = isObject(schema) ? schema : {},
  depth = 0
): JsonSchema {
  if (!isObject(schema) || depth > 12) return { type: "string" }
  if (typeof schema.$ref === "string") {
    const target = resolveRef(schema.$ref, root)
    const rest = { ...schema }
    delete rest.$ref
    return target
      ? toGuidedSchema({ ...target, ...rest }, root, depth + 1)
      : { type: "string" }
  }
  const choices = (schema.anyOf ?? schema.oneOf) as unknown
  if (Array.isArray(choices)) {
    const choice =
      choices.find((c) => isObject(c) && c.type !== "null") ?? choices[0]
    return toGuidedSchema(
      {
        ...(isObject(choice) ? choice : {}),
        ...(schema.description !== undefined && {
          description: schema.description
        })
      },
      root,
      depth + 1
    )
  }
  if (Array.isArray(schema.allOf)) {
    const merged: JsonSchema = { ...schema, allOf: undefined }
    for (const part of schema.allOf) {
      if (!isObject(part)) continue
      Object.assign(merged, part, {
        properties: {
          ...(isObject(merged.properties) ? merged.properties : {}),
          ...(isObject(part.properties) ? part.properties : {})
        },
        required: [
          ...(Array.isArray(merged.required) ? merged.required : []),
          ...(Array.isArray(part.required) ? part.required : [])
        ]
      })
    }
    delete merged.allOf
    return toGuidedSchema(merged, root, depth + 1)
  }

  const out: JsonSchema = {}
  for (const key of KEPT) {
    if (schema[key] !== undefined) out[key] = schema[key]
  }
  if (Array.isArray(out.type)) {
    out.type = out.type.find((t) => t !== "null") ?? "string"
  }
  if (out.type === undefined) {
    if (isObject(schema.properties)) out.type = "object"
    else if (schema.items !== undefined) out.type = "array"
    else if (Array.isArray(out.enum)) out.type = typeof out.enum[0]
    else if (out.const === undefined) out.type = "string"
  }
  if (out.type === "object") {
    const properties = isObject(schema.properties) ? schema.properties : {}
    out.properties = Object.fromEntries(
      Object.entries(properties).map(([key, value]) => [
        key,
        toGuidedSchema(value, root, depth + 1)
      ])
    )
    out.required = Array.isArray(schema.required)
      ? [...new Set(schema.required)].filter(
          (key): key is string => typeof key === "string" && key in properties
        )
      : []
    // Without this the model adds fields of its own.
    out.additionalProperties = false
  }
  if (out.type === "array") {
    out.items = toGuidedSchema(schema.items ?? {}, root, depth + 1)
  }
  return out
}

/** A tool's input schema, or an empty object's when it has none. */
export const inputSchemaOf = (tool: Tool): JsonSchema =>
  toGuidedSchema(
    tool.inputSchema ?? { type: "object", properties: {}, required: [] }
  )

/** A tool's description, to its first paragraph and at most 400 characters. */
function shortDescription(tool: Tool) {
  const first = (tool.description ?? "").trim().split(/\n\s*\n/)[0] ?? ""
  const line = first.replace(/\s+/g, " ")
  return line.length > 400 ? `${line.slice(0, 399)}…` : line
}

/**
 * What the system prompt says about the tools. The decision call reads it,
 * and so does the answer, which sees past calls and results.
 */
export function toolsPrompt(tools: ReadonlyArray<Tool>) {
  return [
    "You can use these tools:",
    ...tools.map((tool) => `- ${tool.name}: ${shortDescription(tool)}`),
    "Use a tool when the user asks for one or the answer needs current or outside information; otherwise answer directly. Once you have a tool's result, answer from it."
  ].join("\n")
}

/**
 * The decision's schema: answer, or one of the tools. `required` leaves
 * answering out (`toolChoice: "required"`).
 */
export function decisionSchema(toolNames: Array<string>, required: boolean) {
  return {
    type: "object",
    properties: {
      action: { enum: required ? toolNames : ["answer", ...toolNames] }
    },
    required: ["action"],
    additionalProperties: false
  }
}

/** The decision's reply, read; undefined if it isn't one. */
export function parseDecision(text: string): Decision | undefined {
  try {
    const value = JSON.parse(text) as unknown
    return isObject(value) && typeof value.action === "string"
      ? { action: value.action }
      : undefined
  } catch {
    return undefined
  }
}

/** A tool call's JSON arguments, or undefined if they aren't a JSON object. */
export function parseArguments(
  json: string
): Record<string, unknown> | undefined {
  try {
    const value = JSON.parse(json || "{}") as unknown
    return isObject(value) ? value : undefined
  } catch {
    return undefined
  }
}

/** A tool call as the model sees it in the conversation, without native tool use. */
export function toolCallText(name: string, args: unknown) {
  return JSON.stringify({ tool: name, arguments: args ?? {} })
}

/** Tools as the browser takes them, for its own tool use. */
export const toLanguageModelTools = (
  tools: ReadonlyArray<Tool>
): Array<LanguageModelTool> =>
  tools.map((tool) => ({
    name: tool.name,
    description: tool.description ?? "",
    inputSchema: inputSchemaOf(tool)
  }))

const nativeSupport = new Map<string, Promise<boolean>>()

/**
 * Whether this browser runs tools itself (the spec's open-loop tool use):
 * it has the tool message classes, and `availability()` takes a session
 * that writes tool calls. Checked once per set of languages.
 */
export function supportsNativeTools(
  languages: ReadonlyArray<string>
): Promise<boolean> {
  const api = languageModel()
  if (!api || !toolClasses()) return Promise.resolve(false)
  const key = languages.join(",")
  let supported = nativeSupport.get(key)
  if (!supported) {
    supported = api
      .availability({
        ...baseOptions(languages, { nativeTools: true }),
        tools: [
          {
            name: "check",
            description: "Checks whether tools work.",
            inputSchema: { type: "object", properties: {} }
          }
        ]
      })
      .then(
        (availability) => availability !== "unavailable",
        () => false
      )
    nativeSupport.set(key, supported)
  }
  return supported
}

/** Turns native tool use off for these languages, after a session refused it. */
export function noNativeTools(languages: ReadonlyArray<string>) {
  nativeSupport.set(languages.join(","), Promise.resolve(false))
}
