import baseConfig from "../../eslint.config.mjs"

export default [
  ...baseConfig,
  {
    // wasm/ holds the compat build's Emscripten glue.
    ignores: ["**/out-tsc", "wasm"]
  }
]
