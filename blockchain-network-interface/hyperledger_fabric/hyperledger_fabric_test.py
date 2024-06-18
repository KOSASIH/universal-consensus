import unittest
from hyperledger_fabric import HyperledgerFabric

class TestHyperledgerFabric(unittest.TestCase):
    def setUp(self):
        self.fabric = HyperledgerFabric()

    def test_load_config(self):
        self.assertIsNotNone(self.fabric.config)

    def test_get_config(self):
        self.assertEqual(self.fabric.get_config('network_profile')['name'], 'my_network')

    def test_deploy_chaincode(self):
        chaincode_path = '/path/to/chaincode'
        chaincode_id = 'my_chaincode'
        chaincode_version = '1.0'
        args = ['init', 'a', 'b']
        response = self.fabric.deploy_chaincode(chaincode_path, chaincode_id, chaincode_version, args)
        self.assertIsNotNone(response)

    def test_invoke_chaincode(self):
        chaincode_id = 'my_chaincode'
        args = ['invoke', 'a', 'b']
        peers = ['peer0.org1.example.com', 'peer1.org1.example.com']
        response = self.fabric.invoke_chaincode(chaincode_id, args, peers)
        self.assertIsNotNone(response)

    def test_query_chaincode(self):
        chaincode_id = 'my_chaincode'
        args = ['query', 'a', 'b']
        peers = ['peer0.org1.example.com', 'peer1.org1.example.com']
        response = self.fabric.query_chaincode(chaincode_id, args, peers)
        self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
