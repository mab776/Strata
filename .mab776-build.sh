#!/bin/bash
# Build the engine + strata_kq (sm_120) inside the strata-server image, at low priority beside the live coder:
#   docker run --rm -v "<tree>:/strata" strata-server:0.1.20-sparse bash /strata/.mab776-build.sh
cd /strata; C=/strata/.venv/bin/cmake
[ -d build ] || $C -G Ninja -DCMAKE_MAKE_PROGRAM=/strata/.venv/bin/ninja -S /strata -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=/strata/third_party/llama.cpp > build-cmake.log 2>&1 || { tail -30 build-cmake.log; exit 1; }
nice -n 19 $C --build build --target strata strata_kq -j 4 > build-ninja.log 2>&1; rc=$?
grep -E "error|FAILED" build-ninja.log | head -30
[ $rc = 0 ] && mkdir -p engine && cp build/strata engine/strata.new && mv -f engine/strata.new engine/strata && cp build/libstrata_kq.so engine/kq.new && mv -f engine/kq.new engine/libstrata_kq.so && echo "ENGINE OK"   # rename: the live engine may be running
chown -R 1000:1000 engine build build-cmake.log build-ninja.log
