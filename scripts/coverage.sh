#!/bin/sh
# usage: scripts/coverage.sh [--quick]
# Prints function / branch-arm coverage of minicelt (see minicelt/test/coverage.ts) to stdout; nothing is
# uploaded or gated. Needs wasm-opt (binaryen) and wasm-dis in PATH, and Deno. The module is built
# unoptimized (functions stay separate; `target/cov` is its own target dir), with a 1 MiB shadow stack
# (the unoptimized frames do not fit the 8 KiB one), then instrumented with `wasm-opt --log-execution`.
set -e
cd "$(dirname "$0")/.."
MINICELT_SHADOW_STACK=1048576 cargo build --profile release -p minicelt-wasm --target wasm32-unknown-unknown \
  --config 'target.wasm32-unknown-unknown.rustflags=["-Ctarget-feature=+simd128"]' \
  --config profile.release.opt-level=0 --config profile.release.lto=false \
  --config profile.release.strip=false --config profile.release.debug=1 --target-dir target/cov
wasm-opt target/cov/wasm32-unknown-unknown/release/minicelt_wasm.wasm -g --log-execution --all-features \
  -o target/cov/minicelt-cov.wasm
deno run --allow-read --allow-run minicelt/test/coverage.ts target/cov/minicelt-cov.wasm test/audio "$@"
