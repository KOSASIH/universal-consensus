// Import necessary libraries and dependencies
use omnia_chain::core::storage::{Storage, StorageId};
use omnia_chain::utils::crypto::{Hash, Signature};
use ipfs_api::IpfsClient;

// Define the StorageNetwork struct
pub struct StorageNetwork {
    // IPFS client
    ipfs: IPFS,
    // Storage nodes
    nodes: Vec<StorageNode>,
}

impl StorageNetwork {
    // Create a new storage network
    pub fn new() -> Self {
        StorageNetwork {
            ipfs: IPFS::new(),
            nodes: vec![],
        }
    }

    // Add a new storage node to the network
    pub fn add_node(&mut self, node: StorageNode) {
        self.nodes.push(node);
    }

    // Remove a storage node from the network
    pub fn remove_node(&mut self, node_id: StorageId) {
        self.nodes.retain(|node| node.id()!= node_id);
    }

    // Get a storage node by ID
    pub fn get_node(&self, node_id: StorageId) -> Option<&StorageNode> {
        self.nodes.iter().find(|node| node.id() == node_id)
    }

    // Get all storage nodes
    pub fn get_nodes(&self) -> Vec<&StorageNode> {
        self.nodes.iter().collect()
    }

    // Add data to the storage network
    pub fn add_data(&mut self, data: Vec<u8>) -> Result<String, String> {
        let hash = self.ipfs.add_data(data)?;
        for node in &mut self.nodes {
            node.add_data(hash.clone(), data.clone())?;
        }
        Ok(hash)
    }

    // Get data from the storage network
    pub fn get_data(&mut self, hash: String) -> Result<Vec<u8>, String> {
        for node in &mut self.nodes {
            match node.get_data(hash.clone()) {
                Ok(data) => return Ok(data),
                Err(error) => continue,
            }
        }
        Err("Data not found".to_string())
    }
}
