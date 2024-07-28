# fabric_integration.py
import os
import json
from fabric_sdk import FabricSDK

class FabricIntegration:
    def __init__(self, config_file='fabric_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def connect_to_fabric(self):
        fabric_sdk = FabricSDK(self.config['fabric_url'], self.config['fabric_username'], self.config['fabric_password'])
        return fabric_sdk

    def deploy_chaincode(self, chaincode_path):
        fabric_sdk = self.connect_to_fabric()
        fabric_sdk.deploy_chaincode(chaincode_path)

    def invoke_chaincode(self, chaincode_id, function, args):
        fabric_sdk = self.connect_to_fabric()
        return fabric_sdk.invoke_chaincode(chaincode_id, function, args)

    def query_chaincode(self, chaincode_id, function, args):
        fabric_sdk = self.connect_to_fabric()
        return fabric_sdk.query_chaincode(chaincode_id, function, args)
