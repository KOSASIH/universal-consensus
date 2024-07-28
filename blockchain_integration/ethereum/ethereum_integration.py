import os
import json
from web3 import Web3

class EthereumIntegration:
    def __init__(self, config_file='ethereum_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def connect_to_ethereum(self):
        w3 = Web3(Web3.HTTPProvider(self.config['ethereum_url']))
        return w3

    def deploy_contract(self, contract_path):
        w3 = self.connect_to_ethereum()
        return w3.eth.deploy_contract(contract_path)

    def call_contract_function(self, contract_address, function_name, args):
        w3 = self.connect_to_ethereum()
        return w3.eth.call_contract_function(contract_address, function_name, args)

    def get_block(self, block_number):
        w3 = self.connect_to_ethereum()
        return w3.eth.get_block(block_number)

    def get_transaction(self, tx_id):
        w3 = self.connect_to_ethereum()
        return w3.eth.get_transaction(tx_id)
