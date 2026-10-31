import baseConfig from "../../eslint.config.mjs"

export default [
  ...baseConfig,
  {
    // wasm/ holds the compat build's Emscripten glue, vendor/ wllama's
    // built JavaScript (scripts/build-wasm.sh).
    ignores: ["**/out-tsc", "wasm", "vendor"]
  }
]
