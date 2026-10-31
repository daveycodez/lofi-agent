// Fails unless each Emscripten glue file after argv[2] is byte-identical to
// the glue embedded in wllama's built JavaScript at argv[2], which is what
// loads those builds.
import { readFileSync } from "node:fs"

const [esm, ...glues] = process.argv.slice(2)
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
const embedded = (0, eval)(source.slice(start, end + 1))
for (const glue of glues) {
  if (readFileSync(glue, "utf8") !== embedded) {
    console.error(`${glue} differs from the glue in ${esm}.`)
    process.exit(1)
  }
}
console.log("The main builds' glue is the one in wllama's JavaScript.")
