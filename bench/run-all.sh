#!/bin/sh
# usage: [RUNTIMES="deno bun"] bench/run-all.sh [passes] > target/bench/results.jsonl   (runs sequentially; build everything first)
cd "$(dirname "$0")/.."
export ROOT=$PWD PATH=$HOME/.deno/bin:$PATH
P=${1:-15}; N=target/bench/native; H=target/bench/host; T=x86_64-unknown-linux-gnu
# RUNTIMES="deno bun" runs only those wasm arms (default: all four); libopus always runs
has() { case " ${RUNTIMES:-deno bun wasmtime wasmer} " in *" $1 "*) return 0;; esac; return 1; }
for codec in silk celt; do for cx in 1 5 9; do
  emit() { printf '{"codec":"%s","cx":%s,"runtime":"%s","build":"%s",' $codec $cx "$1" "$2"; shift 2; "$@" 2>/dev/null | sed 's/^{//' || echo '"error":1}'; }
  for b in scalar avx avx2; do emit libopus $b $N/$codec-$b $codec $cx $P $ROOT/target/bench/pcm$([ $codec = silk ] && echo 16000 || echo 48000).raw; done
  for v in wasm2 wasm3; do
    has deno && emit deno $v deno run -A bench/bench.mjs $codec $v $cx $P
    has bun && emit bun $v bun bench/bench.mjs $codec $v $cx $P
    has wasmtime && emit wasmtime $v $H-wt/$T/release/bench-host $codec $v $cx $P
    has wasmer && emit wasmer $v $H-wr/$T/release/bench-host $codec $v $cx $P
  done
done; done
