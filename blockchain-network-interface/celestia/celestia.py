import os
import json
from blockchain_network_interface import BlockchainNetwork

class Celestia(BlockchainNetwork):
    def __init__(self, config_file='celestia_config.json'):
        self.config = self.load_config(config_file)
        self.network = self.create_network()

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config

    def create_network(self):
        network = BlockchainNetwork()
        network.name = self.config['name']
        network.blockchain_type = self.config['blockchain_type']
        network.peers = self.config['peers']
        network.consensus_algorithm = self.config['consensus_algorithm']
        network.block_time = self.config['block_time']
        network.block_gas_limit = self.config['block_gas_limit']
        network.block_reward = self.config['block_reward']
        return network

    def start_network(self):
        self.network.start()

    def stop_network(self):
        self.network.stop()

if __name__ == '__main__':
    celestia = Celestia()
    celestia.start_network()
