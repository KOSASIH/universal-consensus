// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};
use ipfs_api::IpfsClient;

// Define the IPFS struct
pub struct IPFS {
    // IPFS client
    client: IpfsClient,
}

impl IPFS {
    // Create a new IPFS client
    pub fn new() -> Self {
        IPFS {
            client: IpfsClient::default(),
        }
    }

    // Add data to IPFS
    pub fn add_data(&self, data: Vec<u8>) -> Result<String, String> {
        match self.client.add(data) {
            Ok(result) => Ok(result.hash),
            Err(error) => Err(error.to_string()),
        }
    }

    // Get data from IPFS
    pub fn get_data(&self, hash: String) -> Result<Vec<u8>, String> {
        match self.client.cat(hash) {
            Ok(result) => Ok(result.collect()),
            Err(error) => Err(error.to_string()),
        }
    }
}
