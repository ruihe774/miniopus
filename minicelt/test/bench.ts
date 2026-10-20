// A/B encode-time benchmark for the minicelt wasm builds. Run after `scripts/build.sh celt`; copy the
// builds to compare to other names first (the build overwrites target/minicelt-wasm{2,3}.wasm):
//   deno run -A minicelt/test/bench.ts test/audio old=/path/old.wasm new=target/minicelt-wasm2.wasm
// Encodes all of <audio dir> concatenated (16 kHz mono, x3 upsampled by repetition; the right channel of the
// stereo runs is a delayed, inverted copy of the left) through minicelt_encode_i16, one fresh instance per run,
// REPS (default 15) interleaved runs per configuration, and prints min / median ms per implementation, the
// change against the first one, and the output size (which must match between builds that should be bit-equal).
// Differences below about 0.2 % are noise-level; run nothing else on the machine meanwhile.
const FRAME = 960;
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
const parts = [...Deno.readDirSync(dir)].filter((e) => e.name.endsWith('.wav')).map((e) => e.name).sort().map((n) => readWav(`${dir}/${n}`));
const total = parts.reduce((s, p) => s + p.length, 0);
const L = new Int16Array(total);
let o = 0;
for (const p of parts) { L.set(p, o); o += p.length; }
const R = L.map((_, i) => (i > 9 ? -0.8 * L[i - 9] : 0));
const frames = Math.floor(total / FRAME);

const modules = await Promise.all(impls.map(async (i) => WebAssembly.compile(await Deno.readFile(i.path))));

function run(mod: WebAssembly.Module, ch: number, br: number, cx: number): { ms: number; bytes: number } {
  const e = new WebAssembly.Instance(mod, {}).exports as any;
  if (e.minicelt_init(ch, br, cx, 0, 0) < 0) throw new Error('init');
  const mem = new Int16Array(e.memory.buffer);
  const ptr = e.minicelt_input_i16_ptr() >> 1;
  let bytes = 0;
  const t0 = performance.now();
  for (let f = 0; f < frames; f++) {
    if (ch === 1) mem.set(L.subarray(f * FRAME, (f + 1) * FRAME), ptr);
    else for (let i = 0; i < FRAME; i++) { mem[ptr + 2 * i] = L[f * FRAME + i]; mem[ptr + 2 * i + 1] = R[f * FRAME + i]; }
    const n = e.minicelt_encode_i16();
    if (n < 0) throw new Error('encode ' + n);
    bytes += n;
  }
  return { ms: performance.now() - t0, bytes };
}

const REPS = Number(Deno.env.get('REPS') ?? 15);
console.log(`${frames} frames (${(frames * 0.02).toFixed(1)} s) per run, ${REPS} interleaved runs, min / median ms`);
for (const [ch, br, cx] of [[1, 24000, 10], [1, 64000, 10], [2, 64000, 10], [2, 128000, 10], [1, 64000, 5]]) {
  const ms: number[][] = impls.map(() => []);
  const bytes: number[] = impls.map(() => 0);
  for (let r = -1; r < REPS; r++) {
    impls.forEach((_, k) => {
      const res = run(modules[k], ch, br, cx);
      if (r >= 0) ms[k].push(res.ms);
      bytes[k] = res.bytes;
    });
  }
  const st = ms.map((a) => { const s = [...a].sort((x, y) => x - y); return { min: s[0], med: s[s.length >> 1] }; });
  const base = st[0];
  console.log(`ch=${ch} br=${br} cx=${cx}: ` + impls.map((i, k) =>
    `${i.label} ${st[k].min.toFixed(1)}/${st[k].med.toFixed(1)}` + (k ? ` (${((st[k].min / base.min - 1) * 100).toFixed(2)}% / ${((st[k].med / base.med - 1) * 100).toFixed(2)}%)` : '') + ` ${bytes[k]}B`).join('  |  '));
}
