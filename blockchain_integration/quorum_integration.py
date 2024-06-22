# blockchain_integrations/quorum_integration.py
import os
import json
from quorum_sdk import QuorumSDK

class QuorumIntegration:
    def __init__(self, config_file='quorum_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def connect_to_quorum(self):
        quorum_sdk = QuorumSDK(self.config['quorum_url'], self.config['quorum_username'], self.config['quorum_password'])
        return quorum_sdk

    def deploy_smart_contract(self, contract_path):
        quorum_sdk = self.connect_to_quorum()
        quorum_sdk.deploy_smart_contract(contract_path)

    def invoke_smart_contract(self, contract_id, function, args):
        quorum_sdk = self.connect_to_quorum()
        return quorum_sdk.invoke_smart_contract(contract_id, function, args)

    def query_smart_contract(self, contract_id, function, args):
        quorum_sdk = self.connect_to_quorum()
        return quorum_sdk.query_smart_contract(contract_id, function, args)
