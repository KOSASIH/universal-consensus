// Import necessary libraries and dependencies
use omnia_chain::core::node::{Node, NodeId};
use omnia_chain::utils::crypto::{Hash, Signature};
use omnia_chain::network::{Network, NetworkId};
use omnia_chain::storage::storage_network::{StorageNetwork, StorageNode};

// Define the NodeManager struct
pub struct NodeManager {
    // Node list
    nodes: Vec<Node>,
    // Node network
    network: NetworkId,
    // Node storage network
    storage_network: StorageNetwork,
}

impl NodeManager {
    // Create a new node manager
    pub fn new(network: NetworkId, storage_network: StorageNetwork) -> Self {
        NodeManager {
            nodes: vec![],
            network,
            storage_network,
        }
    }

    // Add a new node to the node list
    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    // Remove a node from the node list
    pub fn remove_node(&mut self, node_id: NodeId) {
        self.nodes.retain(|node| node.id() != node_id);
    }

    // Get a nodeby ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id() == node_id)
    }

    // Get all nodes
    pub fn get_nodes(&self) -> Vec<&Node> {
        self.nodes.iter().collect()
    }

    // Send a message to all nodes
    pub fn broadcast_message(&self, message: Vec<u8>) -> Result<(), String> {
        for node in &self.nodes {
            node.send_message(node, message.clone())?;
        }

        Ok(())
    }

    // Update the node network
    pub fn update_network(&mut self, new_network: NetworkId) {
        self.network = new_network;
    }

    // Update the node storage network
    pub fn update_storage_network(&mut self, new_storage_network: StorageNetwork) {
        self.storage_network = new_storage_network;
    }
}

// Implement the NodeManager's logic
impl NodeManager {
    // Update the node list
    pub fn update_nodes(&mut self, new_nodes: Vec<Node>) {
        self.nodes = new_nodes;
    }

    // Get the node network
    pub fn network(&self) -> NetworkId {
        self.network
    }

    // Get the node storage network
    pub fn storage_network(&self) -> StorageNetwork {
        self.storage_network.clone()
    }
}
