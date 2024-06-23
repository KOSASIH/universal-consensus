import threading
import time
import random
import logging

class UniversalConsensus:
    def __init__(self, nodes, quorum_size, timeout=10, max_retries=3):
        self.nodes = nodes
        self.quorum_size = quorum_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.proposals = {}
        self.votes = {}
        self.consensus_value = None
        self.logger = logging.getLogger(__name__)

    def propose(self, node, value):
        with self.lock:
            if node not in self.proposals:
                self.proposals[node] = value
                self.logger.info(f"Proposal from node {node}: {value}")
                self.cond.notify_all()
            else:
                self.logger.warning(f"Node {node} has already proposed a value")

    def vote(self, node, proposal_node, value):
        with self.lock:
            if proposal_node in self.proposals:
                if node not in self.votes:
                    self.votes[node] = (proposal_node, value)
                    self.logger.info(f"Vote from node {node} for proposal from node {proposal_node}: {value}")
                    self.cond.notify_all()
                else:
                    self.logger.warning(f"Node {node} has already voted")
            else:
                self.logger.warning(f"No proposal from node {proposal_node} to vote on")

    def get_consensus(self, node):
        with self.lock:
            while self.consensus_value is None:
                self.cond.wait(self.timeout)
                if self.timeout_reached():
                    self.logger.error("Timeout reached, no consensus achieved")
                    return None
            return self.consensus_value

    def timeout_reached(self):
        return time.time() - self.start_time > self.timeout

    def run_consensus(self):
        self.start_time = time.time()
        while True:
            with self.lock:
                if len(self.votes) >= self.quorum_size:
                    self.consensus_value = self.determine_consensus()
                    self.logger.info(f"Consensus achieved: {self.consensus_value}")
                    break
                elif self.timeout_reached():
                    self.logger.error("Timeout reached, no consensus achieved")
                    break
                else:
                    self.cond.wait(self.timeout)

    def determine_consensus(self):
        # Simple majority vote for now, can be replaced with more advanced algorithms
        votes = {}
        for node, (proposal_node, value) in self.votes.items():
            if value not in votes:
                votes[value] = 1
            else:
                votes[value] += 1
        max_votes = max(votes.values())
        consensus_value = [value for value, count in votes.items() if count == max_votes][0]
        return consensus_value

    def start(self):
        threading.Thread(target=self.run_consensus).start()

    def stop(self):
        self.lock.acquire()
        self.cond.notify_all()
        self.lock.release()

# Example usage
if __name__ == "__main__":
    nodes = ["Node1", "Node2", "Node3", "Node4", "Node5"]
    quorum_size = 3
    consensus = UniversalConsensus(nodes, quorum_size)
    consensus.start()

    # Propose values from different nodes
    consensus.propose("Node1", "Value1")
    consensus.propose("Node2", "Value2")
    consensus.propose("Node3", "Value3")

    # Vote on proposals
    consensus.vote("Node1", "Node1", "Value1")
    consensus.vote("Node2", "Node1", "Value1")
    consensus.vote("Node3", "Node2", "Value2")
    consensus.vote("Node4", "Node2", "Value2")
    consensus.vote("Node5", "Node3", "Value3")

    # Get consensus value
    print(consensus.get_consensus("Node1"))  # Should print "Value1"
