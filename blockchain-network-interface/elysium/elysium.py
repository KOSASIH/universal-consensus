import os
import json
from blockchain_network_interface import BlockchainNetwork
from galactic_consensus_nexus import GalacticConsensusNexus

class Elysium(BlockchainNetwork):
    def __init__(self, config_file='elysium_config.json'):
        self.config = self.load_config(config_file)
        self.network = self.create_network()
        self.galactic_nexus = GalacticConsensusNexus(self.network)

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
        self.galactic_nexus.start()

    def stop_network(self):
        self.network.stop()
        self.galactic_nexus.stop()

    def connect_to_galactic_nexus(self, network_id):
        self.galactic_nexus.connect_to_network(network_id)

    def disconnect_from_galactic_nexus(self, network_id):
        self.galactic_nexus.disconnect_from_network(network_id)

if __name__ == '__main__':
    elysium = Elysium()
    elysium.start_network()
