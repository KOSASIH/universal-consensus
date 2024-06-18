import os
import json
from decentralized_network_interface import DecentralizedNetwork
from galactic_consensus_nexus import GalacticConsensusNexus
from nova_node import NovaNode
from nova_protocol import NovaProtocol

class NovaSpire(DecentralizedNetwork):
    def __init__(self, config_file='nova_spire_config.json'):
        self.config = self.load_config(config_file)
        self.network = self.create_network()
        self.galactic_nexus = GalacticConsensusNexus(self.network)
        self.nodes = self.create_nodes()
        self.protocol = NovaProtocol(self.network, self.nodes)

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config

    def create_network(self):
        network = DecentralizedNetwork()
        network.name = self.config['name']
        network.network_type = self.config['network_type']
        network.peers = self.config['peers']
        network.consensus_algorithm = self.config['consensus_algorithm']
        network.block_time = self.config['block_time']
        network.block_gas_limit = self.config['block_gas_limit']
        network.block_reward = self.config['block_reward']
        return network

    def create_nodes(self):
        nodes = []
        for node_config in self.config['nodes']:
            node = NovaNode(node_config['host'], node_config['port'])
            nodes.append(node)
        return nodes

    def start_network(self):
        self.network.start()
        self.galactic_nexus.start()
        for node in self.nodes:
            node.start()

    def stop_network(self):
        self.network.stop()
        self.galactic_nexus.stop()
        for node in self.nodes:
            node.stop()

    def connect_to_galactic_nexus(self, network_id):
        self.galactic_nexus.connect_to_network(network_id)

    def disconnect_from_galactic_nexus(self, network_id):
        self.galactic_nexus.disconnect_from_network(network_id)

    def send_transaction(self, transaction):
        self.protocol.send_transaction(transaction)

    def get_network_state(self):
        return self.protocol.get_network_state()

if __name__ == '__main__':
    nova_spire = NovaSpire()
    nova_spire.start_network()
