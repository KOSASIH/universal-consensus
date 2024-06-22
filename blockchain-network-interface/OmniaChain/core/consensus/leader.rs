// Import necessary libraries and dependencies
use omnia_chain::core::consensus::{Consensus, ConsensusResult};
use omnia_chain::utils::crypto::{Hash, Signature};
use omnia_chain::network::{Node, NodeId};
use omnia_chain::storage::storage_network::{StorageNetwork, StorageNode};

// Define the Leader Election consensus algorithm
pub struct Leader {
    // Nodes
    nodes: Vec<Node>,
    // Block time
    block_time: u64,
    // Block reward
    block_reward: u64,
    // Storage network
    storage_network: StorageNetwork,
}

impl Consensus for Leader {
    fn new(nodes: Vec<Node>, block_time: u64, block_reward: u64, storage_network: StorageNetwork) -> Self {
        Leader {
            nodes,
            block_time,
            block_reward,
            storage_network,
        }
    }

    fn propose_block(&mut self, node: &Node) -> ConsensusResult {
        // Check if the node is the leader
        if!self.is_leader(node) {
            return Err("Node is not the leader".to_string());
        }

        // Generate a new block
        let block = self.generate_block(node);

        // Broadcast the block to the network
        self.storage_network.broadcast_block(block.clone());

        Ok(block)
    }

    fn vote(&mut self, node: &Node, block: &Block) -> ConsensusResult {
        // Verify the block
        if!self.verify_block(block) {
            return Err("Block is invalid".to_string());
        }

        // Add the vote to the block
        block.add_vote(node.id);

        // Check if the block has enough votes
        if block.votes.len() >= self.nodes.len() * 2 / 3 {
            // Commit the block to the storage network
            self.storage_network.commit_block(block.clone());
            Ok(block)
        } else {
            Err("Block does not have enough votes".to_string())
        }
    }

    fn verify_block(&self, block: &Block) -> bool {
        // Verify the block's hash and signature
        block.hash() == block.signature().hash() && block.signature().verify(&block.hash())
    }

    fn generate_block(&self, node: &Node) -> Block {
        // Generate a new block with the node's ID and a random nonce
        Block::new(node.id, self.block_time, self.block_reward, rand::random::<u64>())
    }

    fn is_leader(&self, node: &Node) -> bool{
        // Check if the node is the leader
        node.id == self.get_leader()
    }

    fn update_leader(&mut self, new_leader: NodeId) {
        // Update the leader
        self.leader = new_leader;
    }

    fn get_leader(&self) -> NodeId {
        // Get the leader
        self.leader
    }
}

// Implement the Leader algorithm's logic
impl Leader {
    fn update_nodes(&mut self, new_nodes: Vec<Node>) {
        self.nodes = new_nodes;
    }

    fn get_nodes(&self) -> Vec<Node> {
        self.nodes.clone()
    }
}
