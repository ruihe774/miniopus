#!/bin/sh
# Builds three libopus 1.6.1 configurations (all -ffp-contract=off, Release) + bench binaries:
#   scalar  OPUS_DISABLE_INTRINSICS, baseline x86-64
#   avx     intrinsics up to SSE4.1 presumed, no AVX2 kernels, whole library -mavx
#   avx2    AVX2 kernels presumed, whole library -mavx2 -mfma
set -e
cd "$(dirname "$0")/.."
OPUS=$PWD/third_party/opus; out=$PWD/target/bench/native; mkdir -p $out
build() { name=$1; flags=$2; shift 2
  cmake -S $OPUS -B $out/build-$name -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-ffp-contract=off $flags" "$@" >/dev/null
  cmake --build $out/build-$name >/dev/null
  lib=$out/build-$name/libopus.a
  cc -O2 $flags -I$OPUS/include bench/native.c $lib -lm -o $out/silk-$name
  cp $lib $out/build-$name/libopus-noanalysis.a; ar d $out/build-$name/libopus-noanalysis.a analysis.c.o
  cc -O2 $flags -I$OPUS/include -I$OPUS/celt -I$OPUS/src -c minicelt/test/analysis_stub.c -o $out/stub-$name.o
  cc -O2 $flags -I$OPUS/include bench/native.c $out/stub-$name.o $out/build-$name/libopus-noanalysis.a -lm -o $out/celt-$name
}
build scalar "" -DOPUS_DISABLE_INTRINSICS=ON
build avx "-mavx" -DOPUS_X86_MAY_HAVE_AVX2=OFF -DOPUS_X86_PRESUME_SSE4_1=ON
build avx2 "-mavx2 -mfma" -DOPUS_X86_PRESUME_AVX2=ON
