import unittest
from omni_chain import OmniChain

class TestOmniChain(unittest.TestCase):
    def setUp(self):
        self.omni_chain = OmniChain()

    def test_load_config(self):
        self.assertIsNotNone(self.omni_chain.config)

    def test_create_network(self):
        self.assertIsNotNone(self.omni_chain.network)

    def test_create_nodes(self):
        self.assertIsNotNone(self.omni_chain.nodes)

    def test_start_network(self):
        self.omni_chain.start_network()
        self.assertTrue(self.omni_chain.network.is_running())
        self.assertTrue(self.omni_chain.galactic_nexus.is_running())
        for node in self.omni_chain.nodes:
            self.assertTrue(node.is_running())

    def test_stop_network(self):
        self.omni_chain.stop_network()
        self.assertFalse(self.omni_chain.network.is_running())
        self.assertFalse(self.omni_chain.galactic_nexus.is_running())
        for node in self.omni_chain.nodes:
            self.assertFalse(node.is_running())

    def test_connect_to_galactic_nexus(self):
        self.omni_chain.connect_to_galactic_nexus("network1")
        self.assertTrue(self.omni_chain.galactic_nexus.is_connected("network1"))

    def test_disconnect_from_galactic_nexus(self):
        self.omni_chain.disconnect_from_galactic_nexus("network1")
        self.assertFalse(self.omni_chain.galactic_nexus.is_connected("network1"))

    def test_send_transaction(self):
        transaction = {"from": "0x123", "to": "0x456", "value": 1}
        self.omni_chain.send_transaction(transaction)

    def test_get_blockchain_state(self):
        state = self.omni_chain.get_blockchain_state()
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
