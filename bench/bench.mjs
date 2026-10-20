// usage: deno run -A bench.mjs | bun bench.mjs   <silk|celt> <wasm2|wasm3> <complexity> <passes>
// Same protocol as native.c: fresh init per pass (untimed), whole PCM file per pass, JSON line out.
import { readFileSync } from 'node:fs';
const argv = typeof Deno !== 'undefined' ? Deno.args : process.argv.slice(2);
const [codec, variant, cx, passes] = [argv[0], argv[1], +argv[2], +argv[3]];
const silk = codec === 'silk', frame = silk ? 320 : 960, warm = 3;
const root = new URL('..', import.meta.url).pathname;
const wasm = readFileSync(`${root}target/mini${codec}-${variant}.wasm`);
const pcm = readFileSync(`${root}target/bench/pcm${silk ? 16000 : 48000}.raw`);
const s = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length >> 1);
const frames = Math.floor(s.length / frame);
const x = Float32Array.from(s.subarray(0, frames * frame), (v) => v * (1 / 32768));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports, p = `mini${codec}_`;
const init = silk ? () => e.minisilk_init(24000, cx, 0) : () => e.minicelt_init(1, 64000, cx, 1, 0);
const inPtr = e[p + 'input_ptr'](), outPtr = e[p + 'output_ptr'](), enc = e[p + 'encode'];
const t = []; let bytes = 0;
for (let k = -warm; k < passes; k++) {
  if (init() < 0) throw new Error('init');
  let b = 0;
  const t0 = performance.now();
  for (let i = 0; i < frames; i++) {
    // memory may be re-grown only at init; refresh the view each frame cheaply
    new Float32Array(e.memory.buffer, inPtr, frame).set(x.subarray(i * frame, (i + 1) * frame));
    const r = enc(); if (r < 0) throw new Error('encode ' + r);
    b += r;
  }
  const dt = performance.now() - t0;
  if (k >= 0) t.push(dt); bytes = b;
}
t.sort((a, b) => a - b);
console.log(JSON.stringify({ frames, bytes, ms_min: +t[0].toFixed(3), ms_med: +t[passes >> 1].toFixed(3) }));
