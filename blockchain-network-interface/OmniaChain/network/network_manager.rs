// Import necessary libraries and dependencies
use omnia_chain::core::network::{Network, NetworkId};
use omnia_chain::utils::crypto::{Hash, Signature};
use omnia_chain::node::{Node, NodeId};

// Define the NetworkManager struct
pub struct NetworkManager {
    // Network list
    networks: Vec<Network>,
}

impl NetworkManager {
    // Create a new network manager
    pub fn new() -> Self {
        NetworkManager {
            networks: vec![],
        }
    }

    // Add a new network to the network list
    pub fn add_network(&mut self, network: Network) {
        self.networks.push(network);
    }

    // Remove a network from the network list
    pub fn remove_network(&mut self, network_id: NetworkId) {
        self.networks.retain(|network| network.id()!= network_id);
    }

    // Get a network by ID
    pub fn get_network(&self, network_id: NetworkId) -> Option<&Network> {
        self.networks.iter().find(|network| network.id() == network_id)
    }

    // Get all networks
    pub fn get_networks(&self) -> Vec<&Network> {
        self.networks.iter().collect()
    }

    // Update the network list
    pub fn update_networks(&mut self, new_networks: Vec<Network>) {
        self.networks = new_networks;
    }
}

// Implement the NetworkManager's logic
impl NetworkManager {
    // Get the network list
    pub fn networks(&self) -> Vec<&Network> {
        self.networks.iter().collect()
    }
}
