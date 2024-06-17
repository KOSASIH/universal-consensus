// pbft_test.rs
use crate::pbft::{PBFT, ClientRequest, ViewChange};
use std::collections::HashMap;
use std::sync::mpsc::channel;
use std::thread;

#[test]
fn test_pbft_view_change() {
    let nodes = vec!["node1".to_string(), "node2".to_string(), "node3".to_string()];
    let node_id = 0;
    let view = 0;
    let pbft = PBFT::new(nodes, node_id, view);

    let (tx, rx) = channel();
    thread::spawn(move || {
        let _ = pbft.view_change(ViewChange {
            view: 1,
            node_id: 1,
        });
        tx.send(()).unwrap();
    });

    rx.recv().unwrap();
}

#[test]
fn test_pbft_client_request() {
    let nodes = vec!["node1".to_string(), "node2".to_string(), "node3".to_string()];
    let node_id = 0;
    let view = 0;
    let pbft = PBFT::new(nodes, node_id, view);

    let request = ClientRequest {
        client_id: "client1".to_string(),
        request_id: 0,
        operation: "read".to_string(),
    };

    let result = pbft.handle_client_request(request);
    assert!(result.is_ok());
}
