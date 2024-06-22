# corda_integration.py
import os
import json
from corda_sdk import CordaSDK

class CordaIntegration:
    def __init__(self, config_file='corda_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def connect_to_corda(self):
        corda_sdk = CordaSDK(self.config['corda_url'], self.config['corda_username'], self.config['corda_password'])
        return corda_sdk

    def deploy_cor_dapp(self, cor_dapp_path):
        corda_sdk = self.connect_to_corda()
        corda_sdk.deploy_cor_dapp(cor_dapp_path)

    def invoke_flow(self, flow_name, args):
        corda_sdk = self.connect_to_corda()
        return corda_sdk.invoke_flow(flow_name, args)

    def query_state(self, state_name, args):
        corda_sdk = self.connect_to_corda()
        return corda_sdk.query_state(state_name, args)
