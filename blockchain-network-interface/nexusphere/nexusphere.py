import os
import json
from blockchain_interface import Blockchain
from nexusphere_node import NexusphereNode
from nexusphere_protocol import NexusphereProtocol

class Nexusphere(Blockchain):
    def __init__(self, config_file='nexusphere_config.json'):
        self.config = self.load_config(config_file)
        self.blockchain = self.create_blockchain()
        self.nodes = self.create_nodes()
        self.protocol = NexusphereProtocol(self.blockchain, self.nodes)

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config

    def create_blockchain(self):
        blockchain = Blockchain()
        blockchain.name = self.config['name']
        blockchain.blockchain_type = self.config['blockchain_type']
        blockchain.consensus_algorithm = self.config['consensus_algorithm']
        blockchain.block_time = self.config['block_time']
        blockchain.block_gas_limit = self.config['block_gas_limit']
        blockchain.block_reward = self.config['block_reward']
        return blockchain

    def create_nodes(self):
        nodes = []
        for node_config in self.config['nodes']:
            node = NexusphereNode(node_config['host'], node_config['port'])
            nodes.append(node)
        return nodes

    def start_blockchain(self):
        self.blockchain.start()
        for node in self.nodes:
            node.start()

    def stop_blockchain(self):
        self.blockchain.stop()
        for node in self.nodes:
            node.stop()

    def send_transaction(self, transaction):
        self.protocol.send_transaction(transaction)

    def get_blockchain_state(self):
        return self.protocol.get_blockchain_state()

if __name__ == '__main__':
    nexusphere = Nexusphere()
    nexusphere.start_blockchain()
