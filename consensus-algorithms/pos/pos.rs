// pos.rs
use crate::blockchain::{Block, Blockchain};
use crate::transaction::{Transaction, TransactionPool};

pub struct PoS {
    blockchain: Blockchain,
    transaction_pool: TransactionPool,
    validators: Vec<String>,
}

impl PoS {
    pub fn new(validators: Vec<String>) -> Self {
        Self {
            blockchain: Blockchain::new(),
            transaction_pool: TransactionPool::new(),
            validators,
        }
    }

    pub fn validate_transaction(&mut self, transaction: Transaction) -> bool {
        // TO DO: implement transaction validation logic
        true
    }

    pub fn add_block(&mut self, block: Block) -> bool {
        // TO DO: implement block addition logic
        true
    }

    pub fn get_validators(&self) -> &Vec<String> {
        &self.validators
    }
}
