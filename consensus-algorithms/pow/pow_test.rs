// pow_test.rs
use crate::pow::{PoW, Block, Header};
use std::time::Instant;

#[test]
fn test_pow_mine() {
    let pow = PoW::new(10);
    let block = Block {
        header: Header {
            prev_block_hash: vec![0u8; 32],
            transactions: vec![],
            nonce: 0,
        },
        transactions: vec![],
    };

    let start = Instant::now();
    let result = pow.mine(&block);
    let duration = start.elapsed();

    assert!(result.len() > 0);
    println!("Mining took: {:?} seconds", duration);
}

#[test]
fn test_pow_difficulty() {
    let pow = PoW::new(20);
    let block = Block {
        header: Header {
            prev_block_hash: vec![0u8; 32],
            transactions: vec![],
            nonce: 0,
        },
        transactions: vec![],
    };

    let result = pow.mine(&block);
    assert!(result.len() > 0);
    assert!(result.leading_zeros() >= 20);
}
