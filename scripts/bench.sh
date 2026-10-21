#!/bin/sh
# usage: scripts/bench.sh [-e wasmtime|wasmer-cranelift|wasmer-llvm]... [-n REPS] <audio dir> label=path.wasm...
# Builds and runs minicelt/bench, the native A/B encode-time benchmark of minicelt wasm builds (usage
# details in minicelt/bench/src/main.rs). The engines compile the module ahead of time, so the code that
# runs does not depend on JIT tiering. Wasmtime (the default) is the bench backend; the Wasmer engines are kept
# but excluded (minicelt/PORTING.md "Bench backends"). `-e wasmer-llvm` builds the harness with the Wasmer LLVM backend,
# which needs LLVM 22 (llvm-22-dev, libpolly-22-dev); set LLVM_SYS_221_PREFIX if it is not in /usr/lib/llvm-22.
set -e
cd "$(dirname "$0")/.."
features=
for a in "$@"; do [ "$a" = wasmer-llvm ] && features=wasmer-llvm; done
if [ -n "$features" ] && [ -z "$LLVM_SYS_221_PREFIX" ] && [ -x /usr/lib/llvm-22/bin/llvm-config ]; then
  export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22
fi
# --target host-tuple: the workspace's .cargo/config.toml makes wasm32 the default target.
cargo run -q --release --manifest-path minicelt/bench/Cargo.toml --target host-tuple ${features:+--features $features} -- "$@"
