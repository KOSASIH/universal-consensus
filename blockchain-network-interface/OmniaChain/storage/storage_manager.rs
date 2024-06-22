// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};

// Define the StorageManager struct
pub struct StorageManager {
    // Storage list
    storages: Vec<Storage>,
}

impl StorageManager {
    // Create a new storage manager
    pub fn new() -> Self {
        StorageManager {
            storages: vec![],
        }
    }

    // Add a new storage to the storage list
    pub fn add_storage(&mut self, storage: Storage) {
        self.storages.push(storage);
    }

    // Remove a storage from the storage list
    pub fn remove_storage(&mut self, storage_id: StorageId) {
        self.storages.retain(|storage| storage.id()!= storage_id);
    }

    // Get a storage by ID
    pub fn get_storage(&self, storage_id: StorageId) -> Option<&Storage> {
        self.storages.iter().find(|storage| storage.id() == storage_id)
    }

    // Get all storages
    pub fn get_storages(&self) -> Vec<&Storage> {
        self.storages.iter().collect()
    }

    // Update the storage list
    pub fn update_storages(&mut self, new_storages: Vec<Storage>) {
        self.storages = new_storages;
    }
}
