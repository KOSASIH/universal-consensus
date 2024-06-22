// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};

// Define the StorageNode struct
pub struct StorageNode {
    // Storage ID
    id: StorageId,
    // Data
    data: Vec<Vec<u8>>,
}

impl StorageNode {
    // Create a new storage node
    pub fn new(id: StorageId) -> Self {
        StorageNode {
            id,
            data: vec![],
        }
    }

    // Add data to the storage node
    pub fn add_data(&mut self, hash: String, data: Vec<u8>) -> Result<(), String> {
        self.data.push(data);
        Ok(())
    }

    // Get data from the storage node
    pub fn get_data(&self, hash: String) -> Result<Vec<u8>, String> {
        for data in &self.data {
            if hash == Hash::sha256(data) {
                return Ok(data.clone());
            }
        }
        Err("Data not found".to_string())
    }
}
