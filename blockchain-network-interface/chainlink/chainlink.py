import os
import json
from web3 import Web3
from web3.contract import Contract

class Chainlink:
    def __init__(self, config):
        self.config = config
        self.web3 = Web3(Web3.HTTPProvider(self.config['provider_url']))
        self.contract = self._load_contract()

    def _load_contract(self):
        with open(self.config['contract_path'], 'r') as f:
            contract_json = json.load(f)
        return Contract.from_abi(contract_json['abi'], self.web3, contract_json['address'])

    def get_price(self, asset):
        return self.contract.functions.getPrice(asset).call()

    def get_conversion_rate(self, from_asset, to_asset):
        return self.contract.functions.getConversionRate(from_asset, to_asset).call()

    def request_data(self, asset, callback_address):
        return self.contract.functions.requestData(asset, callback_address).transact()

    def fulfill_data(self, request_id, data):
        return self.contract.functions.fulfillData(request_id, data).transact()

    def get_request_status(self, request_id):
        return self.contract.functions.getRequestStatus(request_id).call()
