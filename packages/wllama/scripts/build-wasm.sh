#!/usr/bin/env bash
# Builds wasm/wllama-relaxed.wasm: wllama's main WebAssembly build, for the
# exact @wllama/wllama version this package depends on, with two changes:
#
# - llama.cpp's WebAssembly SIMD kernels switched on. Emscripten tells CMake
#   the processor is "x86", so llama.cpp falls back to its generic C kernels;
#   -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm selects arch/wasm/quants.c instead.
# - Relaxed SIMD: llama.cpp PR #19590 (llama-cpp-relaxed-simd.patch), built
#   with -mrelaxed-simd, for Chrome and Firefox. The client picks this file
#   only where the browser validates relaxed SIMD.
#
# The build is otherwise wllama's own (scripts/docker-compose.yml): the same
# Emscripten, Dawn and CMake options, so the JavaScript glue it emits must be
# byte-identical to the one in @wllama/wllama; the script checks that, since
# the published glue is what loads this file.
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
VERSION="$(node -p "require('$PKG/package.json').dependencies['@wllama/wllama']")"
mkdir -p "$WORK"
echo "Building wllama $VERSION in $WORK"

# wllama and the llama.cpp commit it pins.
rm -rf "$WORK/wllama"
git clone -q --depth 1 --branch "$VERSION" https://github.com/ngxson/wllama.git "$WORK/wllama"
git -C "$WORK/wllama" submodule update -q --init --depth 1 llama.cpp
git -C "$WORK/wllama/llama.cpp" apply --whitespace=nowarn "$HERE/llama-cpp-relaxed-simd.patch"

# -mrelaxed-simd for the main build only (the compat build is Safari's).
python3 -I - "$WORK/wllama/CMakeLists.txt" <<'EOF'
import sys
path = sys.argv[1]
s = open(path).read()
old = "        -sMEMORY64=1\n        -fwasm-exceptions\n    )\n    list(APPEND WLLAMA_LINK_OPTIONS"
if old not in s:
    sys.exit("wllama's CMakeLists.txt changed: update build-wasm.sh")
s = s.replace(old, "        -sMEMORY64=1\n        -fwasm-exceptions\n        -mrelaxed-simd\n    )\n    list(APPEND WLLAMA_LINK_OPTIONS", 1)
open(path, "w").write(s)
EOF

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
BUILD="$WORK/wllama/build"
mkdir -p "$BUILD/emdawn"
curl -sL -o "$BUILD/emdawn.zip" \
  "https://github.com/google/dawn/releases/download/$DAWN_TAG/emdawnwebgpu_pkg-$DAWN_TAG.zip"
python3 -I -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$BUILD/emdawn.zip" "$BUILD/emdawn"

cd "$BUILD"
emcmake cmake .. \
  -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
  -DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON \
  -DEMDAWNWEBGPU_DIR="$BUILD/emdawn/emdawnwebgpu_pkg" \
  -DWLLAMA_TEST_BACKEND=OFF > cmake.log
grep -q "Wasm detected" cmake.log || { echo "llama.cpp didn't select its WebAssembly kernels"; exit 1; }
emmake make wllama -j"$(getconf _NPROCESSORS_ONLN)" > make.log

# The published glue has to load this file unchanged.
node "$HERE/check-glue.mjs" "$BUILD/wllama.js"

cp "$BUILD/wllama.wasm" "$PKG/wasm/wllama-relaxed.wasm"
echo "Wrote $PKG/wasm/wllama-relaxed.wasm ($(wc -c < "$PKG/wasm/wllama-relaxed.wasm") bytes)"
