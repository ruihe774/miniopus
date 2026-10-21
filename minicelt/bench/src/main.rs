//! A/B encode-time benchmark for the minicelt wasm builds, compiled ahead of time by native Wasm engines.
//!
//! Every engine compiles the whole module with its optimizing compiler before the first call (no lazy
//! compilation, no tier-up), so the machine code that runs is a pure function of the module and the engine,
//! unlike under V8, whose tiering decisions depend on timing. Usage (after `scripts/build.sh celt`; copy the
//! builds to compare to other names first, the build overwrites target/minicelt-wasm{2,3}.wasm):
//!
//!   scripts/bench.sh [-e ENGINE]... [-n REPS] test/audio old=/path/old.wasm new=target/minicelt-wasm2.wasm
//!
//! ENGINE is `wasmtime` (Cranelift), `wasmer-cranelift` or `wasmer-llvm` (feature `wasmer-llvm`, which
//! scripts/bench.sh enables when asked for it; needs LLVM 22); repeat `-e` to run several, default `wasmtime`.
//! Wasmer's Cranelift has no tail calls, so it skips the WASM 3.0 variant. Only Wasmtime is a bench backend (with Deno and
//! Bun running `minicelt/test/bench.ts`); the Wasmer engines are kept for investigation but excluded (minicelt/PORTING.md
//! "Bench backends"). Every build runs under every engine, and the first
//! (engine, build) pair is the baseline. The workload is that of `minicelt/test/bench.ts`: all of <audio dir>
//! concatenated (16 kHz mono, x3 upsampled by repetition; the right channel of the stereo runs is a delayed,
//! inverted copy of the left) through `minicelt_encode_i16`, one fresh instance per run, REPS (default 15)
//! interleaved runs per configuration after one warm-up round. It prints min / median ms per implementation,
//! the change against the first one, and the output size (which must match between builds that should be
//! bit-equal). Run nothing else on the machine meanwhile.

use std::time::{Duration, Instant};

const FRAME: usize = 960;

/// One instantiated module: the exports the benchmark calls.
trait Encoder {
    fn init(&mut self, channels: i32, bitrate: i32, complexity: i32) -> i32;
    fn input_i16_ptr(&mut self) -> usize;
    fn write(&mut self, offset: usize, bytes: &[u8]);
    fn encode_i16(&mut self) -> i32;
}

/// A compiled module, instantiated afresh for each run.
trait Compiled {
    fn instantiate(&self) -> Box<dyn Encoder>;
}

#[cfg(feature = "wasmtime")]
mod wt {
    use wasmtime::{Config, Engine, Instance, Memory, Module, OptLevel, Store, TypedFunc};

    pub struct Compiled {
        engine: Engine,
        module: Module,
    }

    pub fn compile(wasm: &[u8]) -> Result<Box<dyn super::Compiled>, String> {
        let mut config = Config::new();
        config
            .cranelift_opt_level(OptLevel::Speed)
            .wasm_simd(true)
            .wasm_relaxed_simd(true)
            .relaxed_simd_deterministic(false)
            .wasm_tail_call(true);
        let engine = Engine::new(&config).expect("wasmtime engine");
        // Explicit AOT: compile to a native artifact, then load that.
        let artifact = engine.precompile_module(wasm).map_err(|e| e.to_string())?;
        // SAFETY: the artifact was produced just above by the same engine.
        let module = unsafe { Module::deserialize(&engine, &artifact) }.expect("wasmtime load");
        Ok(Box::new(Compiled { engine, module }))
    }

    struct Encoder {
        store: Store<()>,
        memory: Memory,
        init: TypedFunc<(i32, i32, i32, i32, i32), i32>,
        input_i16_ptr: TypedFunc<(), i32>,
        encode_i16: TypedFunc<(), i32>,
    }

    impl super::Compiled for Compiled {
        fn instantiate(&self) -> Box<dyn super::Encoder> {
            let mut store = Store::new(&self.engine, ());
            let instance = Instance::new(&mut store, &self.module, &[]).expect("instantiate");
            let memory = instance.get_memory(&mut store, "memory").expect("memory");
            let init = instance
                .get_typed_func(&mut store, "minicelt_init")
                .expect("minicelt_init");
            let input_i16_ptr = instance
                .get_typed_func(&mut store, "minicelt_input_i16_ptr")
                .expect("input ptr");
            let encode_i16 = instance
                .get_typed_func(&mut store, "minicelt_encode_i16")
                .expect("encode");
            Box::new(Encoder {
                store,
                memory,
                init,
                input_i16_ptr,
                encode_i16,
            })
        }
    }

    impl super::Encoder for Encoder {
        fn init(&mut self, channels: i32, bitrate: i32, complexity: i32) -> i32 {
            self.init
                .call(&mut self.store, (channels, bitrate, complexity, 0, 0))
                .expect("init trap")
        }
        fn input_i16_ptr(&mut self) -> usize {
            self.input_i16_ptr.call(&mut self.store, ()).expect("trap") as u32 as usize
        }
        fn write(&mut self, offset: usize, bytes: &[u8]) {
            self.memory
                .write(&mut self.store, offset, bytes)
                .expect("write");
        }
        fn encode_i16(&mut self) -> i32 {
            self.encode_i16
                .call(&mut self.store, ())
                .expect("encode trap")
        }
    }
}

#[cfg(any(feature = "wasmer-cranelift", feature = "wasmer-llvm"))]
mod wr {
    use wasmer::sys::{EngineBuilder, Features};
    use wasmer::{Engine, Instance, Memory, Module, Store, TypedFunction, imports};

    pub struct Compiled {
        engine: Engine,
        module: Module,
    }

    pub fn compile(wasm: &[u8], llvm: bool) -> Result<Box<dyn super::Compiled>, String> {
        let mut features = Features::default();
        features
            .simd(true)
            .relaxed_simd(true)
            .tail_call(true)
            .extended_const(true);
        let builder = if llvm {
            #[cfg(feature = "wasmer-llvm")]
            {
                let mut c = wasmer::sys::LLVM::new();
                c.opt_level(wasmer::sys::LLVMOptLevel::Aggressive);
                EngineBuilder::new(c)
            }
            #[cfg(not(feature = "wasmer-llvm"))]
            panic!("wasmer-llvm: rebuild with --features wasmer-llvm")
        } else {
            #[cfg(feature = "wasmer-cranelift")]
            {
                let mut c = wasmer::sys::Cranelift::new();
                c.opt_level(wasmer::sys::CraneliftOptLevel::Speed);
                EngineBuilder::new(c)
            }
            #[cfg(not(feature = "wasmer-cranelift"))]
            panic!("wasmer-cranelift: rebuild with --features wasmer-cranelift")
        };
        let engine: Engine = builder.set_features(Some(features)).engine().into();
        // Module::new compiles every function before returning; there is no lazy or tiered compilation.
        let module = Module::new(&engine, wasm).map_err(|e| e.to_string())?;
        Ok(Box::new(Compiled { engine, module }))
    }

    struct Encoder {
        store: Store,
        memory: Memory,
        init: TypedFunction<(i32, i32, i32, i32, i32), i32>,
        input_i16_ptr: TypedFunction<(), i32>,
        encode_i16: TypedFunction<(), i32>,
    }

    impl super::Compiled for Compiled {
        fn instantiate(&self) -> Box<dyn super::Encoder> {
            let mut store = Store::new(self.engine.clone());
            let instance =
                Instance::new(&mut store, &self.module, &imports! {}).expect("instantiate");
            let e = &instance.exports;
            let memory = e.get_memory("memory").expect("memory").clone();
            let init = e
                .get_typed_function(&store, "minicelt_init")
                .expect("minicelt_init");
            let input_i16_ptr = e
                .get_typed_function(&store, "minicelt_input_i16_ptr")
                .expect("input ptr");
            let encode_i16 = e
                .get_typed_function(&store, "minicelt_encode_i16")
                .expect("encode");
            Box::new(Encoder {
                store,
                memory,
                init,
                input_i16_ptr,
                encode_i16,
            })
        }
    }

    impl super::Encoder for Encoder {
        fn init(&mut self, channels: i32, bitrate: i32, complexity: i32) -> i32 {
            self.init
                .call(&mut self.store, channels, bitrate, complexity, 0, 0)
                .expect("init trap")
        }
        fn input_i16_ptr(&mut self) -> usize {
            self.input_i16_ptr.call(&mut self.store).expect("trap") as u32 as usize
        }
        fn write(&mut self, offset: usize, bytes: &[u8]) {
            self.memory
                .view(&self.store)
                .write(offset as u64, bytes)
                .expect("write");
        }
        fn encode_i16(&mut self) -> i32 {
            self.encode_i16.call(&mut self.store).expect("encode trap")
        }
    }
}

fn compile(engine: &str, wasm: &[u8]) -> Result<Box<dyn Compiled>, String> {
    match engine {
        #[cfg(feature = "wasmtime")]
        "wasmtime" => wt::compile(wasm),
        #[cfg(any(feature = "wasmer-cranelift", feature = "wasmer-llvm"))]
        "wasmer-cranelift" => wr::compile(wasm, false),
        #[cfg(any(feature = "wasmer-cranelift", feature = "wasmer-llvm"))]
        "wasmer-llvm" => wr::compile(wasm, true),
        _ => panic!(
            "engine {engine} unknown or not enabled (features: wasmtime, wasmer-cranelift, wasmer-llvm)"
        ),
    }
}

/// The samples of the `data` chunk of a 16-bit wav, each repeated three times.
fn read_wav(path: &std::path::Path) -> Vec<i16> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let mut o = 12;
    while o + 8 <= b.len() {
        let size = u32_at(o + 4);
        if &b[o..o + 4] == b"data" {
            let data = &b[o + 8..(o + 8 + size).min(b.len())];
            return data
                .chunks_exact(2)
                .flat_map(|s| [i16::from_le_bytes([s[0], s[1]]); 3])
                .collect();
        }
        o += 8 + size + (size & 1);
    }
    panic!("{}: no data chunk", path.display());
}

/// The input of every frame, as the little-endian bytes `minicelt_encode_i16` reads: `[mono, stereo]`.
fn corpus(dir: &str) -> [Vec<Vec<u8>>; 2] {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    names.sort();
    let l: Vec<i16> = names.iter().flat_map(|p| read_wav(p)).collect();
    let r: Vec<i16> = (0..l.len())
        .map(|i| {
            if i > 9 {
                (-0.8 * l[i - 9] as f64) as i16
            } else {
                0
            }
        })
        .collect();
    let frames = l.len() / FRAME;
    let mono = (0..frames).map(|f| {
        l[f * FRAME..(f + 1) * FRAME]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect()
    });
    let stereo = (0..frames).map(|f| {
        (f * FRAME..(f + 1) * FRAME)
            .flat_map(|i| [l[i].to_le_bytes(), r[i].to_le_bytes()])
            .flatten()
            .collect()
    });
    [mono.collect(), stereo.collect()]
}

fn run(c: &dyn Compiled, input: &[Vec<u8>], ch: i32, br: i32, cx: i32) -> (Duration, usize) {
    let mut e = c.instantiate();
    assert!(e.init(ch, br, cx) >= 0, "init");
    let ptr = e.input_i16_ptr();
    let mut bytes = 0;
    let t0 = Instant::now();
    for frame in input {
        e.write(ptr, frame);
        let n = e.encode_i16();
        assert!(n >= 0, "encode {n}");
        bytes += n as usize;
    }
    (t0.elapsed(), bytes)
}

fn main() {
    let mut engines = Vec::new();
    let mut reps: usize = std::env::var("REPS")
        .ok()
        .map_or(15, |v| v.parse().expect("REPS"));
    let mut pos = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-e" => engines.push(args.next().expect("-e ENGINE")),
            "-n" => reps = args.next().expect("-n REPS").parse().expect("-n REPS"),
            _ => pos.push(a),
        }
    }
    if engines.is_empty() {
        engines.push("wasmtime".into());
    }
    assert!(
        pos.len() >= 2,
        "usage: minicelt-bench [-e ENGINE]... [-n REPS] <audio dir> label=path.wasm..."
    );
    let input = corpus(&pos[0]);
    let builds: Vec<(&str, Vec<u8>)> = pos[1..]
        .iter()
        .map(|a| {
            let (label, path) = a.split_once('=').expect("label=path.wasm");
            (
                label,
                std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}")),
            )
        })
        .collect();
    let mut impls = Vec::new();
    for engine in &engines {
        for (label, wasm) in &builds {
            let name = if engines.len() > 1 {
                format!("{engine}/{label}")
            } else {
                label.to_string()
            };
            let t0 = Instant::now();
            // Wasmer's Cranelift backend has no tail calls, so it cannot compile the WASM 3.0 variant.
            match compile(engine, wasm) {
                Ok(compiled) => {
                    eprintln!(
                        "{name}: compiled in {:.0} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                    impls.push((name, compiled));
                }
                Err(e) => eprintln!("{name}: skipped, cannot compile: {e}"),
            }
        }
    }

    assert!(!impls.is_empty(), "nothing to run");
    let frames = input[0].len();
    println!(
        "{frames} frames ({:.1} s) per run, {reps} interleaved runs, engines: {}, min / median ms",
        frames as f64 * 0.02,
        engines.join(", ")
    );
    // Stereo at 6, 16 and 24 kbps run the downmix-to-mono and stereo-width (`stereo_fade`) paths, which the
    // higher bitrates never reach.
    let configs = [
        (1, 24000, 10),
        (1, 64000, 10),
        (2, 64000, 10),
        (2, 128000, 10),
        (1, 64000, 5),
        (2, 6000, 10),
        (2, 16000, 10),
        (2, 24000, 10),
    ];
    for (ch, br, cx) in configs {
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); impls.len()];
        let mut bytes = vec![0; impls.len()];
        for r in 0..=reps {
            for (k, (_, c)) in impls.iter().enumerate() {
                let (t, n) = run(c.as_ref(), &input[ch as usize - 1], ch, br, cx);
                if r > 0 {
                    ms[k].push(t.as_secs_f64() * 1e3);
                }
                bytes[k] = n;
            }
        }
        let st: Vec<(f64, f64)> = ms
            .iter_mut()
            .map(|a| {
                a.sort_by(f64::total_cmp);
                (a[0], a[a.len() / 2])
            })
            .collect();
        let line: Vec<String> = impls
            .iter()
            .enumerate()
            .map(|(k, (name, _))| {
                let (min, med) = st[k];
                let rel = if k > 0 {
                    format!(
                        " ({:.2}% / {:.2}%)",
                        (min / st[0].0 - 1.0) * 100.0,
                        (med / st[0].1 - 1.0) * 100.0
                    )
                } else {
                    String::new()
                };
                format!("{name} {min:.1}/{med:.1}{rel} {}B", bytes[k])
            })
            .collect();
        println!("ch={ch} br={br} cx={cx}: {}", line.join("  |  "));
    }
}
