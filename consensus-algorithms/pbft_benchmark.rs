// pbft_benchmark.rs
use crate::pbft::{PBFT, ClientRequest};
use criterion::{Benchmark, Criterion};
use std::time::Duration;

fn bench_pbft_view_change(c: &mut Criterion) {
    let nodes = vec!["node1".to_string(), "node2".to_string(), "node3".to_string()];
    let node_id = 0;
    let view = 0;
    let pbft = PBFT::new(nodes, node_id, view);

    let mut group = c.benchmark_group("pbft_view_change");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("view_change", |b| {
        b.iter(|| {
            let _ = pbft.view_change(ViewChange {
                view: 1,
                node_id: 1,
            });
        });
    });
    group.finish();
}

fn bench_pbft_client_request(c: &mut Criterion) {
    let nodes = vec!["node1".to_string(), "node2".to_string(), "node3".to_string()];
    let node_id = 0;
    let view = 0;
    let pbft = PBFT::new(nodes, node_id, view);

    let request = ClientRequest {
        client_id: "client1".to_string(),
        request_id: 0,
        operation: "read".to_string(),
    };

    let mut group = c.benchmark_group("pbft_client_request");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("handle_client_request", |b| {
        b.iter(|| {
            let _ = pbft.handle_client_request(request.clone());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pbft_view_change, bench_pbft_client_request);
criterion_main!(benches);
