import unittest
from paragon_net import ParagonNet

class TestParagonNet(unittest.TestCase):
    def setUp(self):
        self.paragon_net = ParagonNet()

    def test_load_config(self):
        self.assertIsNotNone(self.paragon_net.config)

    def test_create_network(self):
        self.assertIsNotNone(self.paragon_net.network)

    def test_create_nodes(self):
        self.assertIsNotNone(self.paragon_net.nodes)

    def test_start_network(self):
        self.paragon_net.start_network()
        self.assertTrue(self.paragon_net.network.is_running())
        self.assertTrue(self.paragon_net.galactic_nexus.is_running())
        for node in self.paragon_net.nodes:
            self.assertTrue(node.is_running())

    def test_stop_network(self):
        self.paragon_net.stop_network()
        self.assertFalse(self.paragon_net.network.is_running())
        self.assertFalse(self.paragon_net.galactic_nexus.is_running())
        for node in self.paragon_net.nodes:
            self.assertFalse(node.is_running())

    def test_connect_to_galactic_nexus(self):
        self.paragon_net.connect_to_galactic_nexus("network1")
        self.assertTrue(self.paragon_net.galactic_nexus.is_connected("network1"))

    def test_disconnect_from_galactic_nexus(self):
        self.paragon_net.disconnect_from_galactic_nexus("network1")
        self.assertFalse(self.paragon_net.galactic_nexus.is_connected("network1"))

    def test_send_transaction(self):
        transaction = {"from": "0x123", "to": "0x456", "value": 1}
        self.paragon_net.send_transaction(transaction)

    def test_get_blockchain_state(self):
        state = self.paragon_net.get_blockchain_state()
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
