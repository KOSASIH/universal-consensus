# ethereum_test.py
import unittest
from ethereum import Ethereum

class TestEthereum(unittest.TestCase):
    def test_ethereum_new(self):
        ethereum = Ethereum()
        self.assertIsNotNone(ethereum)

    def test_ethereum_mine_block(self):
        ethereum = Ethereum()
        block = ethereum.mine_block("miner_address")
        self.assertIsNotNone(block)

    def test_ethereum_get_balance(self):
        ethereum = Ethereum()
        balance = ethereum.get_balance("account_address")
        self.assertIsNotNone(balance)

if __name__ == '__main__':
    unittest.main()
