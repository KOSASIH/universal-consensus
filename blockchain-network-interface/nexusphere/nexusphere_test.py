import unittest
from nexusphere import Nexusphere

class TestNexusphere(unittest.TestCase):
    def setUp(self):
        self.nexusphere = Nexusphere()

    def test_load_config(self):
        self.assertIsNotNone(self.nexusphere.config)

    def test_create_blockchain(self):
        self.assertIsNotNone(self.nexusphere.blockchain)

    def test_create_nodes(self):
        self.assertIsNotNone(self.nexusphere.nodes)

    def test_start_blockchain(self):
        self.nexusphere.start_blockchain()
        self.assertTrue(self.nexusphere.blockchain.is_running())
        for node in self.nexusphere.nodes:
            self.assertTrue(node.is_running())

    def test_stop_blockchain(self):
        self.nexusphere.stop_blockchain()
        self.assertFalse(self.nexusphere.blockchain.is_running())
        for node in self.nexusphere.nodes:
            self.assertFalse(node.is_running())

    def test_send_transaction(self):
        transaction = {"from": "0x123", "to": "0x456", "value": 1}
        self.nexusphere.send_transaction(transaction)

    def test_get_blockchain_state(self):
        state = self.nexusphere.get_blockchain_state()
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
