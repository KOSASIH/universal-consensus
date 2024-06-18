import unittest
from unittest.mock import patch, MagicMock
from blockchain_network_interface.ethereum import Ethereum
from blockchain_network_interface.bitcoin import Bitcoin
from blockchain_network_interface.hyperledger import Hyperledger

class TestBlockchainNetworkInterface(unittest.TestCase):
    def setUp(self):
        self.ethereum = Ethereum()
        self.bitcoin = Bitcoin()
        self.hyperledger = Hyperledger()

    def test_ethereum_send_transaction(self):
        tx = {"from": "Alice", "to": "Bob", "amount": 10}
        self.ethereum.send_transaction(tx)
        self.assertEqual(self.ethereum.transactions[-1], tx)

    def test_bitcoin_get_balance(self):
        address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        balance = self.bitcoin.get_balance(address)
        self.assertEqual(balance, 100)

    def test_hyperledger_deploy_contract(self):
        contract_code = "pragma solidity ^0.6.0; contract MyContract {... }"
        self.hyperledger.deploy_contract(contract_code)
        self.assertEqual(self.hyperledger.contracts[-1].code, contract_code)

    @patch("blockchain_network_interface.ethereum.Ethereum.send_transaction")
    def test_ethereum_send_transaction_with_mock(self, mock_send_transaction):
        tx = {"from": "Alice", "to": "Bob", "amount": 10}
        self.ethereum.send_transaction(tx)
        mock_send_transaction.assert_called_once_with(tx)

if __name__ == "__main__":
    unittest.main()
