#!/usr/bin/env bash
# Builds this package's copy of wllama: its WebAssembly files and its
# JavaScript, from wllama WLLAMA_VERSION on a newer llama.cpp.
#
# - wasm/wllama-relaxed.wasm: the main build with relaxed SIMD, for Chrome
#   and Firefox. The client picks it only where the browser validates
#   relaxed SIMD.
# - wasm/wllama.wasm: the main build without it, for other browsers.
# - wasm/wllama-compat.wasm and wasm/wllama-compat.js.txt, its glue: the
#   compat build, for Safari. The glue is .txt so bundlers emit it as is.
# - vendor/wllama/: wllama's JavaScript (index.js, with the main builds'
#   glue in it, and its .d.ts files), which the client imports instead of
#   @wllama/wllama, and build-id.js, which identifies this build.
#
# Changes from wllama's own builds:
#
# - llama.cpp at LLAMA_CPP_COMMIT rather than the commit wllama pins, for
#   LFM3 ("lfm3", llama.cpp PR #31252). wllama-llama-cpp-sync.patch ports
#   wllama's C++ to its API: wllama's own sync (its master, at 5aeac37) plus
#   the rerank prompt's new arguments. llama-cpp-wasm64.patch fixes a
#   std::max that doesn't compile with 64-bit memory.
# - llama.cpp's WebAssembly SIMD kernels switched on. Emscripten tells CMake
#   the processor is "x86", so llama.cpp falls back to its generic C kernels;
#   -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm selects arch/wasm/quants.c instead.
# - Relaxed SIMD dot products in those kernels (llama-cpp-relaxed-simd.patch,
#   after llama.cpp PR #19590, plus q6_K and q3_K), built with
#   -mrelaxed-simd for the relaxed build only. Their 7-bit operand only
#   holds 0..127, so they compute the same on ARM and on V8's x64 lowering.
# - Faster model loading: llama.cpp reads tensors in file order
#   (llama-cpp-load-in-file-order.patch) and wllama reads ahead 64 MB blocks
#   (wllama-fast-load.patch).
# - Slot state save and restore (wllama-slot-state.patch, wllama issue
#   #298): Wllama.saveSlotState() and restoreSlotState(), llama-server's own
#   slot save and restore, so the client can keep what the model has read
#   across page loads.
# - WebGPU subgroup shaders only where the browser also has WGSL's
#   subgroup_id (llama-cpp-webgpu-subgroup-id.patch): they use it, so
#   browsers with subgroups but not subgroup_id (Chrome 141) failed to
#   compile them and the model didn't load.
# - WebGPU on Imagination PowerVR GPUs (adapter vendor "img-tec", the Pixel
#   10's) made usable (llama-cpp-webgpu-powervr.patch). Those GPUs read
#   storage bindings declared read_write about 3x slower than read-only
#   ones, and every generic shader declares them read_write. On img-tec
#   only: Q4_K and Q6_K mat-vecs and mat-mats go through new shaders that
#   read the weights and a copy of src1 through read-only bindings; one-token
#   flash attention goes through a new shader (the vector path took ms per
#   call); redundant queue waits are skipped (the scheduler waits ~20 times
#   per token and each costs ms on Android); and command buffers are smaller.
#   350M Q4_K_M on the Pixel: prompt 40 -> 165 tokens/s, generation 2.5 ->
#   25. Other GPUs run the code as before.
# - Context checkpoints at user messages: wllama tokenizes chat prompts
#   before the chat session looks for messages in them
#   (wllama-message-spans.patch), and llama.cpp knows where LFM2's messages
#   start (llama-cpp-lfm2-message-spans.patch). LFM2 and LFM2.5 can't roll
#   their memory back, so without a checkpoint there a new chat rereads the
#   system prompt and tools.
#
# The builds are otherwise wllama's own (scripts/docker-compose.yml): the
# same Emscripten, Dawn and CMake options. The JavaScript is wllama's own
# build (build:worker, tsup, tsc) of the patched tree, with the main builds'
# glue in it; the script checks that both main builds have that glue. The
# compat build's glue ships with it.
#
# Needs git, cmake, python3, node and npm. Emscripten is installed into the
# work directory unless EMSDK_DIR points at an emsdk checkout to reuse.
#
# Usage: scripts/build-wasm.sh [work directory]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
WORK="${1:-$(mktemp -d)}"
EMSDK_VERSION="4.0.20"
WLLAMA_VERSION="3.8.1"
LLAMA_CPP_COMMIT="19cce6919992fdffd92e0ac152caaf1e2b8f254e"
mkdir -p "$WORK"
echo "Building wllama $WLLAMA_VERSION with llama.cpp $LLAMA_CPP_COMMIT in $WORK"

# wllama, and llama.cpp at LLAMA_CPP_COMMIT.
rm -rf "$WORK/wllama"
git clone -q --depth 1 --branch "$WLLAMA_VERSION" https://github.com/ngxson/wllama.git "$WORK/wllama"
git -C "$WORK/wllama" apply "$HERE/wllama-llama-cpp-sync.patch"
# 64 MB read-ahead blocks when loading the model.
git -C "$WORK/wllama" apply "$HERE/wllama-fast-load.patch"
git -C "$WORK/wllama" apply "$HERE/wllama-slot-state.patch"
git -C "$WORK/wllama" apply "$HERE/wllama-message-spans.patch"
LLAMA="$WORK/wllama/llama.cpp"
git -C "$LLAMA" init -q
git -C "$LLAMA" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_COMMIT"
git -C "$LLAMA" checkout -q FETCH_HEAD
git -C "$LLAMA" apply --whitespace=nowarn "$HERE/llama-cpp-relaxed-simd.patch"
git -C "$LLAMA" apply "$HERE/llama-cpp-wasm64.patch"
# Tensors loaded in file order, so wllama's reads are sequential.
git -C "$LLAMA" apply "$HERE/llama-cpp-load-in-file-order.patch"
git -C "$LLAMA" apply "$HERE/llama-cpp-lfm2-message-spans.patch"
# Subgroup shaders only where the browser also has WGSL's subgroup_id.
git -C "$LLAMA" apply "$HERE/llama-cpp-webgpu-subgroup-id.patch"
# PowerVR (img-tec) shaders and fixes.
git -C "$LLAMA" apply "$HERE/llama-cpp-webgpu-powervr.patch"

# Emscripten, at the version wllama builds with.
EMSDK_DIR="${EMSDK_DIR:-$WORK/emsdk}"
if [ ! -d "$EMSDK_DIR" ]; then
  git clone -q --depth 1 https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
fi
"$EMSDK_DIR/emsdk" install "$EMSDK_VERSION" >/dev/null
"$EMSDK_DIR/emsdk" activate "$EMSDK_VERSION" >/dev/null
# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1

# Dawn's WebGPU package, at the tag wllama's build downloads.
DAWN_TAG="$(sed -n 's/.*DAWN_TAG=\(v[0-9.]*\).*/\1/p' "$WORK/wllama/scripts/docker-compose.yml" | head -1)"
DAWN="$WORK/wllama/emdawn"
mkdir -p "$DAWN"
curl -sL -o "$DAWN.zip" \
  "https://github.com/google/dawn/releases/download/$DAWN_TAG/emdawnwebgpu_pkg-$DAWN_TAG.zip"
python3 -I -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$DAWN.zip" "$DAWN"

# build <directory> <cmake options...>
build() {
  local dir="$WORK/wllama/$1"
  shift
  mkdir -p "$dir"
  (
    cd "$dir"
    emcmake cmake .. \
      -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
      -DGGML_WEBGPU=ON \
      -DEMDAWNWEBGPU_DIR="$DAWN/emdawnwebgpu_pkg" \
      -DWLLAMA_TEST_BACKEND=OFF "$@" > cmake.log
    grep -q "Wasm detected" cmake.log || { echo "llama.cpp didn't select its WebAssembly kernels"; exit 1; }
    emmake make wllama -j"$(getconf _NPROCESSORS_ONLN)" > make.log
  )
}

build build-relaxed -DGGML_WEBGPU_JSPI=ON \
  -DCMAKE_C_FLAGS=-mrelaxed-simd -DCMAKE_CXX_FLAGS=-mrelaxed-simd
build build -DGGML_WEBGPU_JSPI=ON
build build-compat -DWLLAMA_COMPAT=ON -DLLAMA_WASM_MEM64=OFF -DGGML_WEBGPU_JSPI=OFF

# wllama's JavaScript, with the main build's glue in it: its build:worker,
# build:tsup (ESM only) and build:typedef steps.
(
  cd "$WORK/wllama"
  npm ci --ignore-scripts --no-audit --no-fund > npm.log
  cp build/wllama.js src/wasm/wllama.js
  ./scripts/build_worker.sh
  npx tsup src/index.ts --format esm --clean > tsup.log
  npx tsc --emitDeclarationOnly --declaration -p tsconfig.build.json
)

# Both main builds load with that JavaScript.
node "$HERE/check-glue.mjs" "$WORK/wllama/esm/index.js" \
  "$WORK/wllama/build-relaxed/wllama.js" "$WORK/wllama/build/wllama.js"

mkdir -p "$PKG/wasm"
cp "$WORK/wllama/build-relaxed/wllama.wasm" "$PKG/wasm/wllama-relaxed.wasm"
cp "$WORK/wllama/build/wllama.wasm" "$PKG/wasm/wllama.wasm"
cp "$WORK/wllama/build-compat/wllama.wasm" "$PKG/wasm/wllama-compat.wasm"
cp "$WORK/wllama/build-compat/wllama.js" "$PKG/wasm/wllama-compat.js.txt"

VENDOR="$PKG/vendor/wllama"
rm -rf "$VENDOR"
mkdir -p "$VENDOR"
# index.js and its types (the worker code's and source-map's aren't imported).
(cd "$WORK/wllama/esm" && find . -name index.js -o -name "*.d.ts" ! -path "./workers-code/*" ! -path "./wasm/*" | cpio -pdm --quiet "$VENDOR")
cp "$WORK/wllama/LICENCE" "$VENDOR/LICENCE"
# Saved slot states are only restored into the build that saved them.
BUILD_ID="$(cat "$PKG"/wasm/* "$VENDOR/index.js" | shasum -a 256 | cut -c1-16)"
printf '// Generated by scripts/build-wasm.sh: identifies this build of wllama.\nexport const BUILD_ID = "%s"\n' "$BUILD_ID" > "$VENDOR/build-id.js"
printf 'export declare const BUILD_ID: string\n' > "$VENDOR/build-id.d.ts"
ls -l "$PKG/wasm" "$VENDOR"
