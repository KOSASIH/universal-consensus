# Node implementation
import threading

class Node:
    def __init__(self, node_id: str, nodes: List[str], consensus_algorithm: str):
        self.node_id = node_id
        self.nodes = nodes
        self.consensus_algorithm = consensus_algorithm
        self.thread = threading.Thread(target=self.run)

    def run(self) -> None:
        # Node run implementation
        pass
