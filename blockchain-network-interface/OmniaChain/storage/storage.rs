// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};

// Define the Storage struct
pub struct Storage {
    // Storage ID
    id: StorageId,
    // Data
    data: Vec<u8>,
}

impl Storage {
    // Create a new storage
    pub fn new(id: StorageId, data: Vec<u8>) -> Self {
        Storage {
            id,
            data,
        }
    }

    // Get the storage ID
    pub fn id(&self) -> StorageId {
        self.id
    }

    // Get the data
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    // Update the data
    pub fn update_data(&mut self, new_data: Vec<u8>) {
        self.data = new_data;
    }
}
