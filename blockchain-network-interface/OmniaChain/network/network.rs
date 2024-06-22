// Import necessary libraries and dependencies
use omnia_chain::core::network::{Network, NetworkId};
use omnia_chain::utils::crypto::{Hash, Signature};
use omnia_chain::node::{Node, NodeId};

// Define the Network struct
pub struct Network {
    // Network ID
    id: NetworkId,
    // Node list
    nodes: Vec<NodeId>,
    // Network topology
    topology: Vec<(NodeId, NodeId)>,
}

impl Network {
    // Create a new network
    pub fn new(id: NetworkId) -> Self {
        Network {
            id,
            nodes: vec![],
            topology: vec![],
        }
    }

    // Add a new node to the network
    pub fn add_node(&mut self, node_id: NodeId) {
        self.nodes.push(node_id);
    }

    // Remove a node from the network
    pub fn remove_node(&mut self, node_id: NodeId) {
        self.nodes.retain(|node| *node!= node_id);
    }

    // Add a new connection between two nodes
    pub fn add_connection(&mut self, node1_id: NodeId, node2_id: NodeId) {
        self.topology.push((node1_id, node2_id));
    }

    // Remove a connection between two nodes
    pub fn remove_connection(&mut self, node1_id: NodeId, node2_id: NodeId) {
        self.topology.retain(|(n1, n2)| (*n1, *n2)!= (node1_id, node2_id) && (*n1, *n2)!= (node2_id, node1_id));
    }

    // Get the network ID
    pub fn id(&self) -> NetworkId {
        self.id
    }

    // Get the node list
    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.clone()
    }

    // Get the network topology
    pub fn topology(&self) -> Vec<(NodeId, NodeId)> {
        self.topology.clone()
    }
}

// Implement the Network's logic
impl Network {
    // Update the network ID
    pub fn update_id(&mut self, new_id: NetworkId) {
        self.id = new_id;
    }

    // Update the node list
    pub fn update_nodes(&mut self, new_nodes: Vec<NodeId>) {
        self.nodes = new_nodes;
    }

    // Update the network topology
    pub fn update_topology(&mut self, new_topology: Vec<(NodeId, NodeId)>) {
        self.topology = new_topology;
    }
}
