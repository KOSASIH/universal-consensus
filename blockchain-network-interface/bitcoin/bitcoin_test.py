# bitcoin_test.py
import unittest
from bitcoin import Bitcoin

class TestBitcoin(unittest.TestCase):
    def test_bitcoin_new(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        bitcoin = Bitcoin(config)
        self.assertIsNotNone(bitcoin)

    def test_bitcoin_mine_block(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        bitcoin = Bitcoin(config)
        block = bitcoin.mine_block([])
        self.assertIsNotNone(block)

    def test_bitcoin_get_balance(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        bitcoin = Bitcoin(config)
        balance = bitcoin.get_balance("account_address")
        self.assertIsNotNone(balance)

    def test_bitcoin_get_new_address(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        bitcoin = Bitcoin(config)
        address = bitcoin.get_new_address()
        self.assertIsNotNone(address)
