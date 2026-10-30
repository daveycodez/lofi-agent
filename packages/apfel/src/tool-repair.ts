/**
 * Tool-call repair. Apple's model picks the right tool but often gets its
 * arguments wrong: it renames fields (`limit` for `numResults`, `site` for
 * `includeDomains`) and drops optional ones it was asked to set. So after the
 * model calls a tool, its arguments are written again with apfel's guided
 * generation, which holds the model to the tool's JSON Schema:
 *
 * 1. For each optional field, a yes/no answer: does the request give it a
 *    value? (Asked for the arguments directly, the model leaves out optional
 *    numbers and arrays even when the request states them.)
 * 2. The arguments, with the fields from step 1 required and the rest left
 *    out. Fields the model already got right (schema name, right type) are
 *    kept as it wrote them.
 *
 * Works from the schema alone, for any tool, MCP tools included. Through
 * `chat()` with search, calendar, file and weather tools, the calls came out
 * right 27 times in 32 with this, against 18 without; the misses left are
 * the model not calling a tool at all, which this can't fix.
 */

type JsonSchema = Record<string, unknown>

/** Keywords guided generation can enforce; apfel rejects others with 400. */
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
 * The schema in the subset guided generation enforces: types, properties,
 * required, items, enum, const and descriptions. Local `$ref`s are inlined,
 * `anyOf`/`oneOf`/type unions become their first non-null choice, and
 * constraints it can't enforce (lengths, patterns, formats…) are dropped.
 */
export function toGuidedSchema(
  schema: unknown,
  root: JsonSchema = isObject(schema) ? schema : {},
  depth = 0
): JsonSchema {
  if (!isObject(schema) || depth > 12) return { type: "string" }
  if (typeof schema.$ref === "string") {
    const target = resolveRef(schema.$ref, root)
    return target
      ? toGuidedSchema(
          { ...target, ...withoutKey(schema, "$ref") },
          root,
          depth + 1
        )
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
    const merged = schema.allOf.reduce<JsonSchema>(
      (all, part) => (isObject(part) ? mergeSchemas(all, part) : all),
      withoutKey(schema, "allOf")
    )
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
      ? schema.required.filter(
          (key): key is string => typeof key === "string" && key in properties
        )
      : []
  }
  if (out.type === "array") {
    out.items = toGuidedSchema(schema.items ?? {}, root, depth + 1)
  }
  return out
}

function withoutKey(schema: JsonSchema, key: string) {
  const rest = { ...schema }
  delete rest[key]
  return rest
}

function mergeSchemas(a: JsonSchema, b: JsonSchema): JsonSchema {
  const merged = { ...a, ...b }
  if (isObject(a.properties) || isObject(b.properties)) {
    merged.properties = {
      ...(isObject(a.properties) ? a.properties : {}),
      ...(isObject(b.properties) ? b.properties : {})
    }
  }
  if (Array.isArray(a.required) || Array.isArray(b.required)) {
    merged.required = [
      ...new Set([
        ...(Array.isArray(a.required) ? a.required : []),
        ...(Array.isArray(b.required) ? b.required : [])
      ])
    ]
  }
  return merged
}

/** Whether `value` fits a guided schema's type and choices. */
function fits(value: unknown, schema: JsonSchema): boolean {
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) return false
  if (schema.const !== undefined) return value === schema.const
  switch (schema.type) {
    case "string":
      return typeof value === "string"
    case "integer":
      return Number.isInteger(value)
    case "number":
      return typeof value === "number" && Number.isFinite(value)
    case "boolean":
      return typeof value === "boolean"
    case "array":
      return (
        Array.isArray(value) &&
        value.every((item) => fits(item, (schema.items ?? {}) as JsonSchema))
      )
    case "object":
      return isObject(value)
    default:
      return true
  }
}

/**
 * The fields of the model's own arguments that already have a schema's
 * name and a value of the right type. They're kept as they are: with several
 * calls to one tool in a reply, they're what tells the calls apart.
 */
function validDraftFields(
  draft: string | undefined,
  properties: Record<string, JsonSchema>
) {
  let parsed: unknown
  try {
    parsed = draft ? JSON.parse(draft) : undefined
  } catch {
    return {}
  }
  if (!isObject(parsed)) return {}
  return Object.fromEntries(
    Object.entries(parsed).filter(
      ([key, value]) =>
        Object.hasOwn(properties, key) &&
        value !== null &&
        fits(value, properties[key] as JsonSchema)
    )
  )
}

/** One line per field: name, type or choices, and description. */
function describeFields(properties: Record<string, JsonSchema>) {
  return Object.entries(properties)
    .map(([key, field]) => {
      const kind = Array.isArray(field.enum)
        ? field.enum.map((v) => JSON.stringify(v)).join(" | ")
        : String(field.type ?? "any")
      const about =
        typeof field.description === "string" ? `: ${field.description}` : ""
      return `- ${key} (${kind})${about}`
    })
    .join("\n")
}

/**
 * The name of the tool the model meant, when it got the name slightly wrong
 * (case, `-` for `_`…). Undefined if no tool matches.
 */
export function matchToolName(name: string, toolNames: Iterable<string>) {
  const names = [...toolNames]
  if (names.includes(name)) return name
  const plain = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "")
  const matches = names.filter((n) => plain(n) === plain(name))
  return matches.length === 1 ? matches[0] : undefined
}

/** A chat completion request apfel answers with JSON matching `schema`. */
export type GuidedRequest = {
  /** Instructions, sent before the conversation. */
  instructions: string
  schema: JsonSchema
  name: string
}

/** Sends a guided request; resolves to the model's JSON, parsed. */
export type GuidedComplete = (request: GuidedRequest) => Promise<unknown>

export type RepairableTool = {
  name: string
  description: string
  inputSchema?: unknown
}

/**
 * The tool call's arguments, written again under the tool's schema.
 * Undefined if the tool takes no arguments or its schema isn't an object
 * (the model's own arguments are used then).
 */
export async function repairToolArguments(
  tool: RepairableTool,
  complete: GuidedComplete,
  /** The model's own arguments, which may use the wrong field names. */
  draft?: string
): Promise<Record<string, unknown> | undefined> {
  const schema = toGuidedSchema(tool.inputSchema)
  const properties = (schema.properties ?? {}) as Record<string, JsonSchema>
  if (schema.type !== "object" || Object.keys(properties).length === 0) {
    return undefined
  }
  const required = schema.required as Array<string>
  const optional = Object.keys(properties).filter(
    (key) => !required.includes(key)
  )
  const about = `the tool ${tool.name}${tool.description ? ` (${tool.description})` : ""}`
  // With several calls in one reply, the draft says which one this is.
  const attempt =
    draft && draft !== "{}"
      ? `\nThe assistant's first attempt at this call, which may use wrong field names: ${draft.slice(0, 1000)}`
      : ""

  let chosen: Array<string> = []
  if (optional.length > 0) {
    const answer = await complete({
      name: "fields",
      instructions: `You are deciding the arguments of a call to ${about}, for the user's latest request. Its fields:\n${describeFields(properties)}\nFor each optional field, answer true only if the conversation states a value for it, otherwise false.`,
      schema: {
        type: "object",
        properties: Object.fromEntries(
          optional.map((key) => [
            key,
            {
              type: "boolean",
              description: `Does the request state a value for ${key}?`
            }
          ])
        ),
        required: optional
      }
    })
    if (isObject(answer))
      chosen = optional.filter((key) => answer[key] === true)
  }

  const kept = validDraftFields(draft, properties)
  const fields = [
    ...new Set([
      ...required,
      ...chosen,
      ...optional.filter((key) => key in kept)
    ])
  ]
  const args = await complete({
    name: tool.name,
    instructions: `Write the arguments for a call to ${about} that does what the user asks in their latest request.${attempt}`,
    schema: {
      type: "object",
      properties: Object.fromEntries(
        fields.map((key) => [key, properties[key]])
      ),
      required: fields
    }
  })
  if (!isObject(args)) return undefined
  // Optional fields left empty weren't really set.
  const written = Object.fromEntries(
    Object.entries(args).filter(
      ([key, value]) =>
        required.includes(key) ||
        !(
          value === "" ||
          value === null ||
          (Array.isArray(value) && value.length === 0)
        )
    )
  )
  return { ...written, ...kept }
}
