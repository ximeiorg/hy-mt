//! Operator-level micro-benchmarks with a proper GPU sync.
//!
//! candle queues GPU work asynchronously, so each case is repeated `iters` times
//! and followed by a single read-back before the timing is taken.
//!
//! This is the tool that produced the numbers in `MERGE.md` (e.g. the GQA
//! `expand().reshape()` path at ~1 GB/s on Metal versus `stack` at memory speed).
//!
//! ```bash
//! cargo run --release --features metal --example micro_bench
//! ```

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

fn bench(name: &str, iters: usize, mut f: impl FnMut() -> Tensor) {
    // warm-up + drain
    let _ = f().sum_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_scalar::<f32>());
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(f());
    }
    let _ = last
        .unwrap()
        .sum_all()
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_scalar::<f32>())
        .unwrap();
    let per = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{name:<32} {per:>9.1} us/op");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = match Device::new_metal(0) {
        Ok(d) => {
            println!("device: metal");
            d
        }
        Err(_) => {
            println!("device: cpu");
            Device::Cpu
        }
    };
    let dtype = DType::F16;
    let (nkv, nh, hd) = (4usize, 16usize, 128usize);

    for ctx in [16usize, 512usize] {
        println!("--- ctx {ctx} ---");
        let k = Tensor::rand(0f32, 1f32, (1, nkv, ctx, hd), &dev)?.to_dtype(dtype)?;
        let v = Tensor::rand(0f32, 1f32, (1, nkv, ctx, hd), &dev)?.to_dtype(dtype)?;
        let k1 = Tensor::rand(0f32, 1f32, (1, nkv, 1, hd), &dev)?.to_dtype(dtype)?;
        let _v1 = Tensor::rand(0f32, 1f32, (1, nkv, 1, hd), &dev)?.to_dtype(dtype)?;
        let q = Tensor::rand(0f32, 1f32, (1, nh, 1, hd), &dev)?.to_dtype(dtype)?;
        let w = Tensor::rand(0f32, 1f32, (2048, 2048), &dev)?.to_dtype(dtype)?;
        let x = Tensor::rand(0f32, 1f32, (1, 1, 2048), &dev)?.to_dtype(dtype)?;
        let krep = k
            .unsqueeze(2)?
            .broadcast_as((1, nkv, 4, ctx, hd))?
            .contiguous()?
            .reshape((1, nh, ctx, hd))?;

        bench("matmul (1,1,2048)@(2048,2048)", 200, || {
            x.broadcast_matmul(&w).unwrap()
        });
        bench("matmul fused gate_up 2048->12288", 200, || {
            let big = Tensor::rand(0f32, 1f32, (2048, 12288), &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            x.broadcast_matmul(&big).unwrap()
        });
        bench("cat append k", 200, || {
            Tensor::cat(&[&k, &k1], 2).unwrap()
        });
        bench("repeat_kv (bcast+contig)", 200, || {
            k.unsqueeze(2)
                .unwrap()
                .broadcast_as((1, nkv, 4, ctx, hd))
                .unwrap()
                .contiguous()
                .unwrap()
        });
        bench("k transpose+contiguous", 200, || {
            k.transpose(2, 3).unwrap().contiguous().unwrap()
        });
        bench("q @ k^T", 200, || {
            q.matmul(&krep.transpose(2, 3).unwrap()).unwrap()
        });
        bench("softmax (1,16,1,ctx)", 200, || {
            candle_nn::ops::softmax_last_dim(&q.matmul(&krep.transpose(2, 3).unwrap()).unwrap()).unwrap()
        });
        let vrep = v
            .unsqueeze(2)
            .unwrap()
            .broadcast_as((1, nkv, 4, ctx, hd))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((1, nh, ctx, hd))
            .unwrap();
        bench("probs @ v", 200, || {
            let s = q.matmul(&krep.transpose(2, 3).unwrap()).unwrap();
            s.matmul(&vrep).unwrap()
        });
        bench("cat 512 rows (pure copy)", 200, || {
            let big = k.clone();
            Tensor::cat(&[&big, &big], 2).unwrap()
        });
    }
    Ok(())
}
