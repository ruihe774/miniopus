// Measures the pseudostack high-water mark (stack::peak()) and the depth of the wasm shadow stack
// (the Rust call stack, which lives at the bottom of linear memory and grows down from address
// STACK_TOP) over worst-case configurations.
//
//   MINISILK_SHADOW_STACK=1048576 cargo build --release -p minisilk-wasm --target wasm32-unknown-unknown --features stack-peak \
//     --config 'target.wasm32-unknown-unknown.rustflags=["-Ctarget-feature=+simd128"]' --target-dir target/stack-peak
//   deno run --allow-read minisilk/test/stack_peak.ts target/stack-peak/wasm32-unknown-unknown/release/minisilk_wasm.wasm [more.wasm ...]
//
// (Add `--profile checked` for the checked build. For WASM 3.0 use -Ctarget-feature=+simd128,+relaxed-simd,+extended-const,+tail-call.) Runs every
// bitrate (including the extremes, OPUS_AUTO and OPUS_BITRATE_MAX) x every complexity x DTX on/off x
// packet loss 0/100 x i16/f32 input over the speech corpus (test/audio), synthetic voiced speech,
// noise at several levels, a chirp, clipped squares/clicks, tones and DC, and prints the peak. The
// pseudostack overflow panics, so if a change raises the peak above GLOBAL_STACK_SIZE, raise it
// temporarily to measure the new peak, then set it to the peak rounded up to 64.

const root = new URL('../..', import.meta.url).pathname;
const FRAME = 320;
const STACK_TOP = 1 << 20; // shadow stack size of the measuring build (MINISILK_SHADOW_STACK=1048576)

let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const noise = (n: number, amp: number) => Float64Array.from({ length: n }, () => amp * (rnd() * 2 - 1));

function readWav(path: string): Float64Array {
  const b = Deno.readFileSync(path);
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let o = 12; o + 8 <= b.length; o += 8 + v.getUint32(o + 4, true) + (v.getUint32(o + 4, true) & 1)) {
    const size = v.getUint32(o + 4, true);
    if (String.fromCharCode(...b.subarray(o, o + 4)) === 'data') {
      return Float64Array.from({ length: size >> 1 }, (_, i) => v.getInt16(o + 8 + 2 * i, true));
    }
  }
  throw new Error(`no data chunk in ${path}`);
}

const signals: [string, Float64Array][] = [];
for (const e of [...Deno.readDirSync(`${root}test/audio`)].filter((e) => e.name.endsWith('.wav')).sort((a, b) => a.name < b.name ? -1 : 1)) {
  signals.push([e.name, readWav(`${root}test/audio/${e.name}`)]);
}
const fs = 16000;
signals.push(['noise-levels', Float64Array.from([20000, 2000, 200, 20, 2].flatMap((a) => [...noise(fs, a)]))]);
signals.push(['chirp', Float64Array.from({ length: 4 * fs }, (_, i) => 12000 * Math.sin(2 * Math.PI * (50 * i / fs + 950 * (i / fs) ** 2)))]);
signals.push(['square+clicks', Float64Array.from({ length: 3 * fs }, (_, i) => i < 1.5 * fs ? (Math.floor(i / 73) % 2 ? 32767 : -32768) : (i % 4001 === 0 ? 32767 : i % 4001 === 7 ? -32768 : 0))]);
signals.push(['tones+dc', Float64Array.from({ length: 3 * fs }, (_, i) => i < 2 * fs ? 16000 * Math.sin(2 * Math.PI * 440 * i / fs) : 5000)]);
signals.push(['silence', new Float64Array(2 * fs)]);

const bitrates = [500, 3000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 40000, 64000, 128000, 510000, 750000, -1000, -1];
let overall = 0, worst = '';
for (const path of Deno.args) {
  const module = await WebAssembly.compile(await Deno.readFile(path));
  let peak = 0, depth = 0;
  for (const bitrate of bitrates) {
    for (let complexity = 0; complexity <= 10; complexity++) {
      for (const [sname, x] of signals) {
        const dtx = (complexity + bitrate) & 1, loss = complexity & 1 ? 100 : 0, float = (complexity & 2) !== 0;
        // deno-lint-ignore no-explicit-any
        const e = (await WebAssembly.instantiate(module, {})).exports as any;
        // Paint the shadow stack region; the lowest changed byte afterwards is the deepest stack use.
        new Uint8Array(e.memory.buffer, 0, STACK_TOP).fill(0xa5);
        e.minisilk_init(bitrate, complexity, dtx);
        e.minisilk_set_packet_loss_perc(loss);
        for (let f = 0; (f + 1) * FRAME <= x.length; f++) {
          if (float) {
            const p = new Float32Array(e.memory.buffer, e.minisilk_input_ptr(), FRAME);
            for (let i = 0; i < FRAME; i++) p[i] = x[f * FRAME + i] / 32768 * (sname === 'chirp' ? 2.5 : 1);
            e.minisilk_encode();
          } else {
            const p = new Int16Array(e.memory.buffer, e.minisilk_input_i16_ptr(), FRAME);
            for (let i = 0; i < FRAME; i++) p[i] = x[f * FRAME + i];
            e.minisilk_encode_i16();
          }
        }
        const mem = new Uint8Array(e.memory.buffer, 0, STACK_TOP);
        let low = 0;
        while (low < STACK_TOP && mem[low] === 0xa5) low++;
        depth = Math.max(depth, STACK_TOP - low);
        const pk = e.minisilk_stack_peak();
        if (pk > peak) { peak = pk; worst = `${path}: bitrate ${bitrate} complexity ${complexity} dtx ${dtx} loss ${loss} ${float ? 'f32' : 'i16'} ${sname}`; }
      }
    }
  }
  console.log(`${path}: pseudostack peak ${peak} bytes, shadow stack depth ${depth} bytes`);
  overall = Math.max(overall, peak);
}
console.log(`overall peak ${overall} bytes (${worst})`);
