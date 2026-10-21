// A/B microbenchmark of mathops::maxabs. The builds need the `bench-maxabs` feature, which exports
// minicelt_bench_maxabs(off, len, iters) (maxabs called `iters` times on input[off..off + len]). Build each side
// with the flags of scripts/build.sh (WASM 3.0: -Ctarget-feature=+simd128,+relaxed-simd,+extended-const,+tail-call):
//   cargo build --release -p minicelt-wasm --features bench-maxabs --target wasm32-unknown-unknown --target-dir target/bench-maxabs \
//     --config 'target.wasm32-unknown-unknown.rustflags=["-Ctarget-feature=+simd128", "-Cllvm-args=-inline-threshold=0"]'
//   wasm-opt target/bench-maxabs/wasm32-unknown-unknown/release/minicelt_wasm.wasm -O2 -Os -c -lmu --strip-debug \
//     --strip-dwarf --strip-producers --strip-target-features -o /path/new.wasm
//   deno run -A minicelt/test/bench_maxabs.ts test/audio old=/path/old.wasm new=/path/new.wasm
// The input buffer holds 1920 samples of the speech corpus (16 kHz mono, x3 upsampled by repetition, from the
// loudest 40 ms; the right channel a delayed, inverted copy, as in bench.ts), scaled to +-1. The cases are the
// slices the encoder passes: the whole frame (opus layer) and the frame minus the overlap plus the overlap (CELT),
// mono and stereo. Each case runs about SAMPLES (default 2e7) samples per timed call, REPS (default 15) interleaved
// runs after one warm-up, and prints min / median ns per call, the change against the first build, and checks that
// all builds return the same maximum.
const FRAME = 960;
const OVERLAP = 120;
const dir = Deno.args[0];
const impls = Deno.args.slice(1).map((a) => { const [label, path] = a.split('='); return { label, path }; });

function readWav(path: string): Int16Array {
  const b = Deno.readFileSync(path);
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let o = 12; o + 8 <= b.length; o += 8 + v.getUint32(o + 4, true) + (v.getUint32(o + 4, true) & 1)) {
    const size = v.getUint32(o + 4, true);
    if (String.fromCharCode(...b.subarray(o, o + 4)) === 'data') {
      const n = size >> 1;
      return Int16Array.from({ length: 3 * n }, (_, i) => v.getInt16(o + 8 + 2 * Math.floor(i / 3), true));
    }
  }
  throw new Error(path);
}
const name = [...Deno.readDirSync(dir)].filter((e) => e.name.endsWith('.wav')).map((e) => e.name).sort()[0];
const L = readWav(`${dir}/${name}`);
// the 2-frame window with the most energy
let best = 0, bestE = -1;
for (let s = 0; s + 2 * FRAME <= L.length; s += FRAME) {
  let e = 0;
  for (let i = s; i < s + 2 * FRAME; i++) e += L[i] * L[i];
  if (e > bestE) { bestE = e; best = s; }
}
const mono = Float32Array.from(L.subarray(best, best + 2 * FRAME), (v) => v / 32768);
const stereo = new Float32Array(2 * FRAME);
for (let i = 0; i < FRAME; i++) { stereo[2 * i] = mono[i]; stereo[2 * i + 1] = i > 9 ? -0.8 * mono[i - 9] : 0; }

const cases = [
  { name: 'mono frame', input: mono, off: 0, len: FRAME },
  { name: 'mono N-ov', input: mono, off: 0, len: FRAME - OVERLAP },
  { name: 'mono ov', input: mono, off: FRAME - OVERLAP, len: OVERLAP },
  { name: 'stereo frame', input: stereo, off: 0, len: 2 * FRAME },
  { name: 'stereo N-ov', input: stereo, off: 0, len: 2 * (FRAME - OVERLAP) },
  { name: 'stereo ov', input: stereo, off: 2 * (FRAME - OVERLAP), len: 2 * OVERLAP },
];

const inst = await Promise.all(impls.map(async (i) => {
  // deno-lint-ignore no-explicit-any
  const e = new WebAssembly.Instance(await WebAssembly.compile(await Deno.readFile(i.path)), {}).exports as any;
  if (!e.minicelt_bench_maxabs) throw new Error(`${i.path}: no minicelt_bench_maxabs (build with --features bench-maxabs)`);
  return e;
}));

const SAMPLES = Number(Deno.env.get('SAMPLES') ?? 2e7);
const REPS = Number(Deno.env.get('REPS') ?? 15);
console.log(`input from ${name}, ~${SAMPLES} samples per timed call, ${REPS} interleaved runs, min / median ns per maxabs call`);
let ok = true;
for (const c of cases) {
  const iters = Math.ceil(SAMPLES / c.len);
  const ns: number[][] = impls.map(() => []);
  const res: number[] = [];
  for (let r = -1; r < REPS; r++) {
    inst.forEach((e, k) => {
      new Float32Array(e.memory.buffer).set(c.input, e.minicelt_input_ptr() >> 2);
      const t0 = performance.now();
      res[k] = e.minicelt_bench_maxabs(c.off, c.len, iters);
      const t = performance.now() - t0;
      if (r >= 0) ns[k].push(t * 1e6 / iters);
    });
  }
  const st = ns.map((a) => { const s = [...a].sort((x, y) => x - y); return { min: s[0], med: s[s.length >> 1] }; });
  const same = res.every((v) => Object.is(v, res[0]));
  ok &&= same;
  console.log(`${c.name.padEnd(12)} len=${String(c.len).padStart(4)}: ` + impls.map((i, k) =>
    `${i.label} ${st[k].min.toFixed(1)}/${st[k].med.toFixed(1)}` +
    (k ? ` (${((st[k].min / st[0].min - 1) * 100).toFixed(1)}% / ${((st[k].med / st[0].med - 1) * 100).toFixed(1)}%)` : '')).join('  |  ') +
    (same ? '' : `  RESULTS DIFFER: ${res.join(' ')}`));
}
if (!ok) Deno.exit(1);
