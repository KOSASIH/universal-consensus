// dpos_benchmark.rs
use crate::dpos::{DPOS, Validators};
use criterion::{Benchmark, Criterion};
use std::time::Duration;

fn bench_dpos_vote_for_validator(c: &mut Criterion) {
    let mut dpos = DPOS::new(Validators::new(vec!["validator1".to_string(), "validator2".to_string()]));

    let mut group = c.benchmark_group("dpos_vote_for_validator");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("vote_for_validator", |b| {
        b.iter(|| {
            dpos.vote_for_validator("validator1");
        });
    });
    group.finish();
}

fn bench_dpos_add_block(c: &mut Criterion) {
    let mut dpos = DPOS::new(Validators::new(vec!["validator1".to_string(), "validator2".to_string()]));
    let block = Block {
        // TO DO: implement block creation logic
    };

    let mut group = c.benchmark_group("dpos_add_block");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("add_block", |b| {
        b.iter(|| {
            let _ = dpos.add_block(block.clone());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_dpos_vote_for_validator, bench_dpos_add_block);
criterion_main!(benches);
