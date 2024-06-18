import os
import json
from hfc.fabric import Client

class HyperledgerFabric:
    def __init__(self, config_file='hyperledger_fabric_config.json'):
        self.config = self.load_config(config_file)
        self.client = Client(net_profile=self.config['network_profile'])

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config

    def get_config(self, key=None):
        if key is None:
            return self.config
        else:
            return self.config.get(key)

    def deploy_chaincode(self, chaincode_path, chaincode_id, chaincode_version, args):
        request = {
            'chaincodePath': chaincode_path,
            'chaincodeId': chaincode_id,
            'chaincodeVersion': chaincode_version,
            'args': args
        }
        response = self.client.deploy(request)
        return response

    def invoke_chaincode(self, chaincode_id, args, peers):
        request = {
            'chaincodeId': chaincode_id,
            'args': args,
            'peers': peers
        }
        response = self.client.invoke(request)
        return response

    def query_chaincode(self, chaincode_id, args, peers):
        request = {
            'chaincodeId': chaincode_id,
            'args': args,
            'peers': peers
        }
        response = self.client.query(request)
        return response

if __name__ == '__main__':
    fabric = HyperledgerFabric()
    chaincode_path = '/path/to/chaincode'
    chaincode_id = 'my_chaincode'
    chaincode_version = '1.0'
    args = ['init', 'a', 'b']
    response = fabric.deploy_chaincode(chaincode_path, chaincode_id, chaincode_version, args)
    print(response)
