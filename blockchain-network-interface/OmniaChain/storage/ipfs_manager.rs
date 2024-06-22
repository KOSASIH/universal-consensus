// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};
use ipfs_api::IpfsClient;

// Define the IPFSManager struct
pub struct IPFSManager {
    // IPFS client
    ipfs: IPFS,
    // Storage list
    storages: Vec<Storage>,
}

impl IPFSManager {
    // Create a new IPFS manager
    pub fn new() -> Self {
        IPFSManager {
            ipfs: IPFS::new(),
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

    // Add data to IPFS and update the storage list
    pub fn add_data(&mut self, data: Vec<u8>) -> Result<String, String> {
        let hash = self.ipfs.add_data(data)?;
        let storage = Storage::new(hash, data);
        self.add_storage(storage);
        Ok(hash)
    }

    // Get data from IPFS and update the storage list
    pub fn get_data(&mut self, hash: String) -> Result<Vec<u8>, String> {
        match self.ipfs.get_data(hash) {
            Ok(data) => {
                let storage = Storage::new(hash, data);
                self.add_storage(storage);
                Ok(data)
            },
            Err(error) => Err(error),
        }
    }

    // Update the storage list
    pub fn update_storages(&mut self, new_storages: Vec<Storage>) {
        self.storages = new_storages;
    }
}
