# fabric_config.py
import os
import json
from fabric_sdk_py import FabricClient, FabricNetwork

class FabricConfig:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        config_file = os.path.join(os.path.dirname(__file__), 'fabric_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config

    def get_network(self):
        network_config = self.config['network']
        return FabricNetwork(
            network_config['name'],
            network_config['orderer_url'],
            network_config['ca_url'],
            network_config['msp_id'],
            network_config['admin_cert'],
            network_config['admin_key']
        )

    def get_client(self):
        client_config = self.config['client']
        return FabricClient(
            client_config['username'],
            client_config['org_name'],
            client_config['channel_name'],
            client_config['chaincode_name'],
            self.get_network()
        )

    def get_channel(self):
        channel_config = self.config['channel']
        return channel_config['name']

    def get_chaincode(self):
        chaincode_config = self.config['chaincode']
        return chaincode_config['name']

# Example fabric_config.json file
{
    "network": {
        "name": "my-network",
        "orderer_url": "grpcs://orderer.example.com:7050",
        "ca_url": "https://ca.example.com:7054",
        "msp_id": "Org1MSP",
        "admin_cert": "path/to/admin-cert.pem",
        "admin_key": "path/to/admin-key.pem"
    },
    "client": {
        "username": "admin",
        "org_name": "Org1",
        "channel_name": "my-channel",
        "chaincode_name": "my-chaincode"
    },
    "channel": {
        "name": "my-channel"
    },
    "chaincode": {
        "name": "my-chaincode"
    }
}
