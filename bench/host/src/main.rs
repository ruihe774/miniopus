// usage: bench-host-<rt> <silk|celt> <wasm2|wasm3> <complexity> <passes>   (cwd-independent; paths via ROOT env)
use std::time::Instant;

trait Inst {
    fn func(&mut self, name: &str) -> usize;
    fn callf(&mut self, f: usize, args: &[i32]) -> i32;
    fn call(&mut self, name: &str, args: &[i32]) -> i32 { let f = self.func(name); self.callf(f, args) }
    fn write_f32(&mut self, ptr: u32, data: &[f32]);
}

#[cfg(feature = "wasmtime")]
mod rt {
    use super::Inst;
    use wasmtime::*;
    pub struct W { store: Store<()>, inst: Instance, mem: Memory, fs: Vec<Func> }
    pub fn load(bytes: &[u8]) -> W {
        let mut c = Config::new();
        c.wasm_simd(true).wasm_relaxed_simd(true).wasm_tail_call(true).wasm_extended_const(true);
        let engine = Engine::new(&c).unwrap();
        let module = Module::new(&engine, bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let inst = Instance::new(&mut store, &module, &[]).unwrap();
        let mem = inst.get_memory(&mut store, "memory").unwrap();
        W { store, inst, mem, fs: vec![] }
    }
    impl Inst for W {
        fn func(&mut self, name: &str) -> usize {
            let f = self.inst.get_func(&mut self.store, name).unwrap_or_else(|| panic!("no export {name}"));
            self.fs.push(f); self.fs.len() - 1
        }
        fn callf(&mut self, f: usize, args: &[i32]) -> i32 {
            let f = self.fs[f];
            let a: Vec<Val> = args.iter().map(|&v| Val::I32(v)).collect();
            let mut r = [Val::I32(0)];
            f.call(&mut self.store, &a, &mut r).unwrap();
            r[0].unwrap_i32()
        }
        fn write_f32(&mut self, ptr: u32, d: &[f32]) {
            let m = self.mem.data_mut(&mut self.store);
            let dst = &mut m[ptr as usize..ptr as usize + d.len() * 4];
            for (o, v) in dst.chunks_exact_mut(4).zip(d) { o.copy_from_slice(&v.to_le_bytes()); }
        }
    }
}

#[cfg(feature = "wasmer")]
mod rt {
    use super::Inst;
    use wasmer::*;
    use wasmer::sys::{Cranelift, EngineBuilder, Features};
    pub struct W { store: Store, inst: Instance, mem: Memory, fs: Vec<Function> }
    pub fn load(bytes: &[u8]) -> W {
        let mut features = Features::new();
        features.simd(true).relaxed_simd(true).tail_call(true).extended_const(true);
        let compiler = Cranelift::new();
        let engine: Engine = EngineBuilder::new(compiler).set_features(Some(features)).engine().into();
        let mut store = Store::new(engine);
        let module = Module::new(&store, bytes).unwrap();
        let inst = Instance::new(&mut store, &module, &imports! {}).unwrap();
        let mem = inst.exports.get_memory("memory").unwrap().clone();
        W { store, inst, mem, fs: vec![] }
    }
    impl Inst for W {
        fn func(&mut self, name: &str) -> usize {
            let f = self.inst.exports.get_function(name).unwrap_or_else(|_| panic!("no export {name}")).clone();
            self.fs.push(f); self.fs.len() - 1
        }
        fn callf(&mut self, f: usize, args: &[i32]) -> i32 {
            let f = self.fs[f].clone();
            let a: Vec<Value> = args.iter().map(|&v| Value::I32(v)).collect();
            f.call(&mut self.store, &a).unwrap()[0].unwrap_i32()
        }
        fn write_f32(&mut self, ptr: u32, d: &[f32]) {
            let view = self.mem.view(&self.store);
            let bytes: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
            view.write(ptr as u64, &bytes).unwrap();
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (codec, variant, cx, passes) = (a[1].as_str(), a[2].as_str(), a[3].parse::<i32>().unwrap(), a[4].parse::<usize>().unwrap());
    let silk = codec == "silk";
    let frame = if silk { 320 } else { 960 };
    let root = std::env::var("ROOT").unwrap();
    let wasm = std::fs::read(format!("{root}/target/mini{codec}-{variant}.wasm")).unwrap();
    let pcm = std::fs::read(format!("{root}/target/bench/pcm{}.raw", if silk { 16000 } else { 48000 })).unwrap();
    let frames = pcm.len() / 2 / frame;
    let x: Vec<f32> = (0..frames * frame).map(|i| i16::from_le_bytes([pcm[2 * i], pcm[2 * i + 1]]) as f32 * (1.0 / 32768.0)).collect();
    let mut w = rt::load(&wasm);
    let p = format!("mini{codec}_");
    let (mut t, mut bytes) = (Vec::new(), 0i64);
    let finit = w.func(if silk { "minisilk_init" } else { "minicelt_init" });
    let fenc = w.func(&format!("{p}encode"));
    let inptr = w.call(&format!("{p}input_ptr"), &[]) as u32;
    for k in -3i32..passes as i32 {
        let r = if silk { w.callf(finit, &[24000, cx, 0]) } else { w.callf(finit, &[1, 64000, cx, 1, 0]) };
        assert!(r >= 0, "init {r}");
        let mut b = 0i64;
        let t0 = Instant::now();
        for i in 0..frames {
            w.write_f32(inptr, &x[i * frame..(i + 1) * frame]);
            let r = w.callf(fenc, &[]);
            assert!(r >= 0, "encode {r}");
            b += r as i64;
        }
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if k >= 0 { t.push(dt); }
        bytes = b;
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{{\"frames\":{frames},\"bytes\":{bytes},\"ms_min\":{:.3},\"ms_med\":{:.3}}}", t[0], t[passes / 2]);
}
