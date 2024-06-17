// consensus-algorithms/pow/pow.rs
use crypto::hash::Hash;
use crypto::digest::Digest;

pub struct PoW {
    difficulty: u32,
}

impl PoW {
    pub fn new(difficulty: u32) -> Self {
        PoW { difficulty }
    }

    pub fn mine(&self, block: &Block) -> Vec<u8> {
        let mut nonce = 0;
        let mut hash = Hash::new();

        loop {
            let header = block.header.clone();
            header.nonce = nonce;
            let data = header.encode();

            hash.update(&data);
            let digest = hash.finalize();

            if digest.leading_zeros() >= self.difficulty {
                return digest.to_vec();
            }

            nonce += 1;
        }
    }
}
