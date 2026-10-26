#!/bin/bash
# Build Niko's DeepSeek-V4.1 engine (strata-dsv41, with our prompt lookup) in its own build dir; the Qwen engine's build/ and
# engine/ are not touched.  docker run --rm -v "<tree>:/strata" strata-server:0.1.20-sparse bash /strata/.mab776-build-dsv41.sh
cd /strata; C=/strata/.venv/bin/cmake
[ -d build-dsv41 ] || $C -G Ninja -DCMAKE_MAKE_PROGRAM=/strata/.venv/bin/ninja -S /strata -B build-dsv41 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=/strata/third_party/llama.cpp > build-dsv41-cmake.log 2>&1 || { tail -30 build-dsv41-cmake.log; exit 1; }
$C --build build-dsv41 --target strata-dsv41 -j "${JOBS:-30}" > build-dsv41-ninja.log 2>&1; rc=$?
grep -E "error|FAILED" build-dsv41-ninja.log | head -30
[ $rc = 0 ] && mkdir -p engine-dsv41 && cp build-dsv41/strata-dsv41 engine-dsv41/strata-dsv41.new && mv -f engine-dsv41/strata-dsv41.new engine-dsv41/strata-dsv41 && echo "DSV41 OK"
chown -R 1000:1000 engine-dsv41 build-dsv41 build-dsv41-*.log
