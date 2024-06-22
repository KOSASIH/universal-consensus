// Import necessary libraries and dependencies
use omnia_chain::core::crypto::{Hash, Signature};
use omnia_chain::utils::math::{BigUint, BigInt};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

// Define the WalletManager struct
pub struct WalletManager {
    // Wallets
    wallets: HashMap<String, Wallet>,
}

impl WalletManager {
    // Create a new wallet manager
    pub fn new() -> Self {
        WalletManager {
            wallets: HashMap::new(),
        }
    }

    // Create a new wallet
    pub fn create_wallet(&mut self, id: String, private_key: Vec<u8>, public_key: Vec<u8>) -> &Wallet {
        let wallet = Wallet::new(id.clone(), private_key, public_key);
        self.wallets.insert(id, wallet);
        self.wallets.get(&id).unwrap()
    }

    // Get a wallet by ID
    pub fn get_wallet(&self, id: &String) -> Option<&Wallet> {
        self.wallets.get(id)
    }

    // Get all wallets
    pub fn get_all_wallets(&self) -> Vec<&Wallet> {
        self.wallets.values().collect()
    }

    // Add a new transaction to a wallet's transaction history
    pub fn add_transaction(&mut self, wallet_id: &String, transaction: Transaction) {
        if let Some(wallet) = self.wallets.get_mut(wallet_id) {
            wallet.add_transaction(transaction);
        }
    }

    // Update a wallet's balance
    pub fn update_balance(&mut self, wallet_id: &String, amount: BigUint) {
        if let Some(wallet) = self.wallets.get_mut(wallet_id) {
            wallet.update_balance(amount);
        }
    }
}
