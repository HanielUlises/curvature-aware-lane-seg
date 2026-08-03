#!/usr/bin/env bash
# Stage an ONNX Runtime "distribution" out of the pip wheel, for the C++ build to link.
#
# The wheel ships a working runtime — including the CUDA and TensorRT providers, built
# against whatever CUDA the wheel targets — but it is laid out for Python, not for a
# linker: no headers, and no unversioned soname symlink. Meanwhile the standalone C++
# release archives are self-consistent but pin a CUDA and cuDNN major version, so the one
# that matches the machine's CUDA is often not the one whose headers are to hand.
#
# This bridges the two without copying half a gigabyte: the libraries are symlinked from
# the wheel, so the providers still resolve next to the runtime that dlopens them, and
# the headers come from any release archive of a version no *newer* than the wheel's.
# ONNX Runtime's C API is backwards compatible — a binary compiled against the 1.18
# headers requests API version 18 and a 1.23 library serves it — so the pairing is
# supported rather than a trick.
#
#   deploy/tools/link_pip_onnxruntime.sh <headers-dir> [output-dir]
#
#     headers-dir  include/ of an unpacked onnxruntime release archive
#     output-dir   where to stage it (default: deploy/third_party/onnxruntime)
#
# Then configure against the result:
#
#   cmake -S deploy -B deploy/build \
#       -DONNXRUNTIME_INCLUDE_DIR=deploy/third_party/onnxruntime/include \
#       -DONNXRUNTIME_LIBRARY=deploy/third_party/onnxruntime/lib/libonnxruntime.so

set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,28p' "$0"
  exit 2
fi

headers=$1
root=${2:-$(cd "$(dirname "$0")/../.." && pwd)/deploy/third_party/onnxruntime}

if [ ! -f "$headers/onnxruntime_cxx_api.h" ]; then
  echo "no onnxruntime_cxx_api.h in $headers" >&2
  exit 1
fi

capi=$(python3 -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')
runtime=$(ls "$capi"/libonnxruntime.so.* 2>/dev/null | grep -v '\.so\.[0-9]*$' | head -1)
if [ -z "$runtime" ]; then
  runtime=$(ls "$capi"/libonnxruntime.so.* | head -1)
fi
if [ -z "$runtime" ]; then
  echo "no libonnxruntime.so.* in $capi; is onnxruntime-gpu installed?" >&2
  exit 1
fi

mkdir -p "$root/lib" "$root/include"

# The soname is libonnxruntime.so.1, which the wheel has no symlink for; without it the
# loader cannot satisfy the NEEDED entry the linker writes.
ln -sfn "$runtime" "$root/lib/libonnxruntime.so.1"
ln -sfn libonnxruntime.so.1 "$root/lib/libonnxruntime.so"
for provider in "$capi"/libonnxruntime_providers_*.so; do
  [ -e "$provider" ] && ln -sfn "$provider" "$root/lib/$(basename "$provider")"
done
for header in "$headers"/*.h; do
  ln -sfn "$header" "$root/include/$(basename "$header")"
done
[ -d "$headers/core" ] && ln -sfn "$headers/core" "$root/include/core"

echo "staged $root"
echo "  runtime  $(basename "$runtime")"
echo "  headers  $headers"
echo
echo "cmake -S deploy -B deploy/build \\"
echo "    -DONNXRUNTIME_INCLUDE_DIR=$root/include \\"
echo "    -DONNXRUNTIME_LIBRARY=$root/lib/libonnxruntime.so"
