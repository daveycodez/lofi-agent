#!/usr/bin/env bash
# Builds this package's WebAssembly files: wllama's builds, for the exact
# @wllama/wllama version this package depends on, on a newer llama.cpp.
#
# - wasm/wllama-relaxed.wasm: the main build with relaxed SIMD, for Chrome
#   and Firefox. The client picks it only where the browser validates
#   relaxed SIMD.
# - wasm/wllama.wasm: the main build without it, for other browsers.
# - wasm/wllama-compat.wasm and wasm/wllama-compat.js.txt, its glue: the
#   compat build, for Safari. The glue is .txt so bundlers emit it as is.
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
#
# The builds are otherwise wllama's own (scripts/docker-compose.yml): the
# same Emscripten, Dawn and CMake options. The main builds' JavaScript glue
# must be byte-identical to the one in @wllama/wllama, which is what loads
# them; the script checks that. The compat build's glue ships with it.
#
# Needs git, cmake, python3 and node. Emscripten is installed into the work
# directory unless EMSDK_DIR points at an emsdk checkout to reuse.
#
# Usage: scripts/build-wasm.sh [work directory]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
WORK="${1:-$(mktemp -d)}"
EMSDK_VERSION="4.0.20"
LLAMA_CPP_COMMIT="19cce6919992fdffd92e0ac152caaf1e2b8f254e"
VERSION="$(node -p "require('$PKG/package.json').dependencies['@wllama/wllama']")"
mkdir -p "$WORK"
echo "Building wllama $VERSION with llama.cpp $LLAMA_CPP_COMMIT in $WORK"

# wllama, and llama.cpp at LLAMA_CPP_COMMIT.
rm -rf "$WORK/wllama"
git clone -q --depth 1 --branch "$VERSION" https://github.com/ngxson/wllama.git "$WORK/wllama"
git -C "$WORK/wllama" apply "$HERE/wllama-llama-cpp-sync.patch"
# 64 MB read-ahead blocks when loading the model.
git -C "$WORK/wllama" apply "$HERE/wllama-fast-load.patch"
LLAMA="$WORK/wllama/llama.cpp"
git -C "$LLAMA" init -q
git -C "$LLAMA" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_COMMIT"
git -C "$LLAMA" checkout -q FETCH_HEAD
git -C "$LLAMA" apply --whitespace=nowarn "$HERE/llama-cpp-relaxed-simd.patch"
git -C "$LLAMA" apply "$HERE/llama-cpp-wasm64.patch"
# Tensors loaded in file order, so wllama's reads are sequential.
git -C "$LLAMA" apply "$HERE/llama-cpp-load-in-file-order.patch"

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

# The published glue has to load the main builds unchanged.
node "$HERE/check-glue.mjs" "$WORK/wllama/build-relaxed/wllama.js"
node "$HERE/check-glue.mjs" "$WORK/wllama/build/wllama.js"

mkdir -p "$PKG/wasm"
cp "$WORK/wllama/build-relaxed/wllama.wasm" "$PKG/wasm/wllama-relaxed.wasm"
cp "$WORK/wllama/build/wllama.wasm" "$PKG/wasm/wllama.wasm"
cp "$WORK/wllama/build-compat/wllama.wasm" "$PKG/wasm/wllama-compat.wasm"
cp "$WORK/wllama/build-compat/wllama.js" "$PKG/wasm/wllama-compat.js.txt"
ls -l "$PKG/wasm"
