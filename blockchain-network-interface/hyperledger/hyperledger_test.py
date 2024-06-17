# hyperledger_test.py
import unittest
from hyperledger import Hyperledger

class TestHyperledger(unittest.TestCase):
    def test_hyperledger_new(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        hyperledger = Hyperledger(config)
        self.assertIsNotNone(hyperledger)

    def test_hyperledger_deploy_chaincode(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        hyperledger = Hyperledger(config)
        chaincode = "example_chaincode"
        result = hyperledger.deploy_chaincode(chaincode)
        self.assertIsNotNone(result)

    def test_hyperledger_invoke_chaincode(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        hyperledger = Hyperledger(config)
        chaincode = "example_chaincode"
        args = ["arg1", "arg2"]
        result = hyperledger.invoke_chaincode(chaincode, args)
        self.assertIsNotNone(result)

    def test_hyperledger_query_chaincode(self):
        config = json.loads('{"network_id": 1, "chain_id": 1, "gas_limit": 8000000, "block_time": 15, "difficulty": 131072, "reward": 3.0, "min_gas_price": 20}')
        hyperledger = Hyperledger(config)
        chaincode = "example_chaincode"
        args = ["arg1", "arg2"]
        result = hyperledger.query_chaincode(chaincode, args)
        self.assertIsNotNone(result)
