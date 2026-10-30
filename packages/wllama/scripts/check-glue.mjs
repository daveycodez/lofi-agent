// Fails unless the Emscripten glue at argv[2] is byte-identical to the glue
// embedded in the installed @wllama/wllama, which is what loads our build.
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"

const require = createRequire(import.meta.url)
const esm = join(
  dirname(require.resolve("@wllama/wllama/package.json")),
  "esm/index.js"
)
const source = readFileSync(esm, "utf8")
const marker = "WLLAMA_EMSCRIPTEN_CODE = "
const start = source.indexOf(marker) + marker.length
const quote = source[start]
let end = start + 1
for (let escaped = false; end < source.length; end++) {
  const c = source[end]
  if (escaped) escaped = false
  else if (c === "\\") escaped = true
  else if (c === quote) break
}
const published = (0, eval)(source.slice(start, end + 1))
const ours = readFileSync(process.argv[2], "utf8")
if (published !== ours) {
  console.error(
    "The built glue differs from @wllama/wllama's: the build doesn't match the published one."
  )
  process.exit(1)
}
console.log("Glue matches @wllama/wllama.")
