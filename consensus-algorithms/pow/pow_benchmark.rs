// pow_benchmark.rs
use crate::pow::{PoW, Block, Header};
use criterion::{Benchmark, Criterion};
use std::time::Duration;

fn bench_pow_mine(c: &mut Criterion) {
    let pow = PoW::new(10);
    let block = Block {
        header: Header {
            prev_block_hash: vec![0u8; 32],
            transactions: vec![],
            nonce: 0,
        },
        transactions: vec![],
    };

    let mut group = c.benchmark_group("pow_mine");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("mine", |b| {
        b.iter(|| pow.mine(&block));
    });
    group.finish();
}

criterion_group!(benches, bench_pow_mine);
criterion_main!(benches);
