// pos_benchmark.rs
use crate::pos::{PoS, Validators};
use criterion::{Benchmark, Criterion};
use std::time::Duration;

fn bench_pos_validate_transaction(c: &mut Criterion) {
    let mut pos = PoS::new(vec!["validator1".to_string(), "validator2".to_string()]);
    let transaction = Transaction {
        // TO DO: implement transaction creation logic
    };

    let mut group = c.benchmark_group("pos_validate_transaction");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("validate_transaction", |b| {
        b.iter(|| {
            let _ = pos.validate_transaction(transaction.clone());
        });
    });
    group.finish();
}

fn bench_pos_add_block(c: &mut Criterion) {
    let mut pos = PoS::new(vec!["validator1".to_string(), "validator2".to_string()]);
    let block = Block {
        // TO DO: implement block creation logic
    };

    let mut group = c.benchmark_group("pos_add_block");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("add_block", |b| {
        b.iter(|| {
            let _ = pos.add_block(block.clone());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pos_validate_transaction, bench_pos_add_block);
criterion_main!(benches);
