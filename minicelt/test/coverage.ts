// Function and branch-arm coverage of minicelt for the CI log (nothing is uploaded or gated).
//   scripts/coverage.sh        (builds the instrumented module, then runs this; needs wasm-opt, wasm-dis, Deno)
//   deno run -A minicelt/test/coverage.ts <instrumented.wasm> <audio dir> [--quick]
// <instrumented.wasm> is an unoptimized build (so functions are not inlined away) that `wasm-opt --log-execution`
// has instrumented: it calls the import `env.log_execution(id)` at every function entry, loop header, branch arm
// and return. The ids are mapped back to functions through `wasm-dis`' text (names come from the name section,
// v0-mangled), so the report is per function and per logging point, not per source line.
// The workload is the speech corpus encoded in mono and stereo over a spread of bitrates, complexities and ctls
// (not `compare.ts`: that one drives several builds and the native oracle). `--quick` uses a quarter of the audio.

const FRAME = 960;
const [wasmPath, dir] = Deno.args;
const quick = Deno.args.includes('--quick');

function readWav(path: string): Int16Array {
  const b = Deno.readFileSync(path);
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let o = 12; o + 8 <= b.length; o += 8 + v.getUint32(o + 4, true) + (v.getUint32(o + 4, true) & 1)) {
    const size = v.getUint32(o + 4, true);
    if (String.fromCharCode(...b.subarray(o, o + 4)) === 'data') {
      const n = size >> 1;
      return Int16Array.from({ length: 3 * n }, (_, i) => v.getInt16(o + 8 + 2 * Math.floor(i / 3), true)); // 16 kHz -> 48 kHz by repetition
    }
  }
  throw new Error(`no data chunk in ${path}`);
}
const parts = [...Deno.readDirSync(dir)].filter((e) => e.name.endsWith('.wav')).map((e) => e.name).sort().map((n) => readWav(`${dir}/${n}`));
const L = new Int16Array(parts.reduce((s, p) => s + p.length, 0));
let off = 0;
for (const p of parts) { L.set(p, off); off += p.length; }
const R = L.map((_, i) => (i > 9 ? -0.8 * L[i - 9] : 0));
const frames = Math.floor(L.length / FRAME / (quick ? 4 : 1));

// [channels, bitrate, complexity, constrained VBR, DTX, packet loss %, bandwidth (0 = auto)]
const CONFIGS = [
  [1, 24000, 10, 0, 0, 0, 0], [1, 64000, 10, 0, 0, 0, 0], [2, 64000, 10, 0, 0, 0, 0], [2, 128000, 10, 0, 0, 0, 0],
  [1, 64000, 5, 0, 0, 0, 0], [2, 6000, 10, 0, 0, 0, 0], [2, 16000, 10, 0, 0, 0, 0], [2, 24000, 10, 0, 0, 0, 0],
  [2, 32000, 10, 0, 0, 0, 0], [1, 6000, 10, 0, 0, 0, 0], [2, 510000, 10, 0, 0, 0, 0], [1, 64000, 0, 0, 0, 0, 0],
  [2, 64000, 2, 0, 0, 0, 0], [2, 64000, 8, 0, 0, 0, 0], [2, 32000, 10, 1, 0, 0, 0], [1, 16000, 10, 1, 1, 0, 0],
  [2, 64000, 10, 0, 0, 20, 0], [1, 24000, 10, 0, 0, 0, 1101], [2, 64000, 10, 0, 0, 0, 1103],
];

const hits = new Map<number, number>();
const module = await WebAssembly.compile(await Deno.readFile(wasmPath));
for (const [ch, br, cx, vbrc, dtx, loss, bw] of CONFIGS) {
  // deno-lint-ignore no-explicit-any
  const e = new WebAssembly.Instance(module, { env: { log_execution: (id: number) => hits.set(id, (hits.get(id) ?? 0) + 1) } }).exports as any;
  if (e.minicelt_init(ch, br, cx, vbrc, dtx) < 0) throw new Error('minicelt_init');
  e.minicelt_set_packet_loss_perc(loss);
  if (bw) e.minicelt_set_bandwidth(bw);
  const mem = new Int16Array(e.memory.buffer);
  const ptr = e.minicelt_input_i16_ptr() >> 1;
  for (let f = 0; f < frames; f++) {
    if (ch === 1) mem.set(L.subarray(f * FRAME, (f + 1) * FRAME), ptr);
    else for (let i = 0; i < FRAME; i++) { mem[ptr + 2 * i] = L[f * FRAME + i]; mem[ptr + 2 * i + 1] = R[f * FRAME + i]; }
    if (e.minicelt_encode_i16() < 0) throw new Error('minicelt_encode_i16');
  }
}

// logging point ids per function, from the disassembly
const wat = new TextDecoder().decode((await new Deno.Command('wasm-dis', { args: [wasmPath], stdout: 'piped' }).output()).stdout);
const funcs = new Map<string, number[]>();
const logName = wat.match(/\(import "env" "log_execution" \(func (\$\S+)/)?.[1]; // `$fimport$0` unless the name section names it
if (!logName) throw new Error('the module has no log_execution import: not instrumented');
let cur: number[] | undefined;
let afterCall = false;
for (const line of wat.split('\n')) {
  const f = line.match(/^ \(func \$(\S+)/);
  if (f) { cur = []; funcs.set(f[1], cur); afterCall = false; continue; }
  const c = line.match(/^\s+\(i32\.const (\d+)\)/);
  if (c && afterCall && cur) cur.push(Number(c[1]));
  afterCall = line.includes(`(call ${logName}`);
}

// v0 mangling: path components are <decimal length><identifier> ('minicelt::celt::bands::haar1' etc.), between
// back-references (`B<base62>_`) and disambiguators (`s<base62>_`) that must not be read as lengths.
function demangle(n: string): string[] {
  const out: string[] = [];
  for (let i = n.startsWith('_R') ? 2 : 0; i < n.length;) {
    const skip = n.slice(i).match(/^[Bs][0-9a-zA-Z]*_/);
    if (skip) { i += skip[0].length; continue; }
    const m = n.slice(i).match(/^(\d+)/);
    if (m) {
      const len = Number(m[1]), name = n.slice(i + m[1].length, i + m[1].length + len);
      if (len > 0 && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) { out.push(name); i += m[1].length + len; continue; }
    }
    i++;
  }
  return out;
}

type Row = { name: string; module: string; points: number; hit: number };
const rows: Row[] = [];
for (const [mangled, ids] of funcs) {
  const path = demangle(mangled);
  if (path[0] !== 'minicelt' || ids.length === 0) continue; // the crate's own functions (not core, not the exports)
  const module = path.length > 3 ? path[2] : path[1]; // minicelt::celt::<file>::fn, minicelt::opus::<file>::fn
  rows.push({ name: path.slice(1).join('::'), module: path[1] === 'opus' ? `opus/${module}` : module, points: ids.length, hit: ids.filter((i) => (hits.get(i) ?? 0) > 0).length });
}
const pct = (a: number, b: number) => (b ? (100 * a / b).toFixed(1) : '-').padStart(5) + '%';
const entered = rows.filter((r) => r.hit > 0);
const sum = (rs: Row[], k: 'points' | 'hit') => rs.reduce((s, r) => s + r[k], 0);
console.log(`minicelt coverage: ${CONFIGS.length} configurations x ${frames} frames of the speech corpus (function / logging-point level)`);
console.log(`  functions entered   ${entered.length} / ${rows.length} (${pct(entered.length, rows.length).trim()})`);
console.log(`  logging points hit  ${sum(rows, 'hit')} / ${sum(rows, 'points')} (${pct(sum(rows, 'hit'), sum(rows, 'points')).trim()}) (function entries, loop headers, branch arms, returns)`);
console.log('\nper module:');
const mods = [...new Set(rows.map((r) => r.module))].sort();
for (const m of mods) {
  const rs = rows.filter((r) => r.module === m), en = rs.filter((r) => r.hit > 0);
  console.log(`  ${m.padEnd(16)} functions ${String(en.length).padStart(3)}/${String(rs.length).padEnd(3)} points ${pct(sum(rs, 'hit'), sum(rs, 'points'))}`);
}
const never = rows.filter((r) => r.hit === 0).sort((a, b) => a.name < b.name ? -1 : 1);
console.log(`\nnever entered (${never.length}):`);
for (const r of never) console.log(`  ${r.name}`);
const partial = rows.filter((r) => r.hit > 0 && r.hit < r.points).sort((a, b) => a.name < b.name ? -1 : 1);
console.log(`\nentered, with logging points not hit (${partial.length}; a few points in straight-line functions can never be hit, so read these as hints):`);
for (const r of partial) console.log(`  ${r.name.padEnd(56)} ${r.hit}/${r.points}`);
