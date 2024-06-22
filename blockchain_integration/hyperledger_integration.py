# blockchain_integrations/hyperledger_integration.py
import os
import json
from hyperledger_sdk import HyperledgerSDK

class HyperledgerIntegration:
    def __init__(self, config_file='hyperledger_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def connect_to_hyperledger(self):
        hyperledger_sdk = HyperledgerSDK(self.config['hyperledger_url'], self.config['hyperledger_username'], self.config['hyperledger_password'])
        return hyperledger_sdk

    def deploy_chaincode(self, chaincode_path):
        hyperledger_sdk = self.connect_to_hyperledger()
        hyperledger_sdk.deploy_chaincode(chaincode_path)

    def invoke_chaincode(self, chaincode_id, function, args):
        hyperledger_sdk = self.connect_to_hyperledger()
        return hyperledger_sdk.invoke_chaincode(chaincode_id, function, args)

    def query_chaincode(self, chaincode_id, function, args):
        hyperledger_sdk = self.connect_to_hyperledger()
        return hyperledger_sdk.query_chaincode(chaincode_id, function, args)
