// Import necessary libraries and dependencies
use omnia_chain::core::crypto::{Hash, Signature};
use omnia_chain::utils::math::{BigUint, BigInt};
use serde::{Serialize, Deserialize};

// Define the Wallet struct
pub struct Wallet {
    // Wallet ID
    id: String,
    // Private key
    private_key: Vec<u8>,
    // Public key
    public_key: Vec<u8>,
    // Balance
    balance: BigUint,
    // Transaction history
    transaction_history: Vec<Transaction>,
}

impl Wallet {
    // Create a new wallet
    pub fn new(id: String, private_key: Vec<u8>, public_key: Vec<u8>) -> Self {
        Wallet {
            id,
            private_key,
            public_key,
            balance: BigUint::from(0u8),
            transaction_history: vec![],
        }
    }

    // Get the wallet ID
    pub fn id(&self) -> &String {
        &self.id
    }

    // Get the private key
    pub fn private_key(&self) -> &Vec<u8> {
        &self.private_key
    }

    // Get the public key
    pub fn public_key(&self) -> &Vec<u8> {
        &self.public_key
    }

    // Get the balance
    pub fn balance(&self) -> &BigUint {
        &self.balance
    }

    // Get the transaction history
    pub fn transaction_history(&self) -> &Vec<Transaction> {
        &self.transaction_history
    }

    // Add a new transaction to the transaction history
    pub fn add_transaction(&mut self, transaction: Transaction) {
        self.transaction_history.push(transaction);
    }

    // Update the balance
    pub fn update_balance(&mut self, amount: BigUint) {
        self.balance += amount;
    }
}

// Define the Transaction struct
#[derive(Serialize, Deserialize)]
pub struct Transaction {
    // Transaction ID
    id: String,
    // Sender's wallet ID
    sender: String,
    // Receiver's wallet ID
    receiver: String,
    // Amount
    amount: BigUint,
    // Timestamp
    timestamp: u64,
    // Signature
    signature: Vec<u8>,
}

impl Transaction {
    // Create a new transaction
    pub fn new(id: String, sender: String, receiver: String, amount: BigUint, timestamp: u64, signature: Vec<u8>) -> Self {
        Transaction {
            id,
            sender,
            receiver,
            amount,
            timestamp,
            signature,
        }
    }

    // Get the transaction ID
    pub fn id(&self) -> &String {
        &self.id
    }

    // Get the sender's wallet ID
    pub fn sender(&self) -> &String {
        &self.sender
    }

    // Get the receiver's wallet ID
    pub fn receiver(&self) -> &String {
        &self.receiver
    }

    // Get the amount
    pub fn amount(&self) -> &BigUint {
        &self.amount
    }

    // Get the timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    // Get the signature
    pub fn signature(&self) -> &Vec<u8> {
        &self.signature
    }
}
