import unittest
from luminari import Luminari

class TestLuminari(unittest.TestCase):
    def setUp(self):
        self.luminari = Luminari()

    def test_load_config(self):
        self.assertIsNotNone(self.luminari.config)

    def test_create_network(self):
        self.assertIsNotNone(self.luminari.network)

    def test_create_nodes(self):
        self.assertIsNotNone(self.luminari.nodes)

    def test_start_network(self):
        self.luminari.start_network()
        self.assertTrue(self.luminari.network.is_running())
        self.assertTrue(self.luminari.galactic_nexus.is_running())
        for node in self.luminari.nodes:
            self.assertTrue(node.is_running())

    def test_stop_network(self):
        self.luminari.stop_network()
        self.assertFalse(self.luminari.network.is_running())
        self.assertFalse(self.luminari.galactic_nexus.is_running())
        for node in self.luminari.nodes:
            self.assertFalse(node.is_running())

    def test_connect_to_galactic_nexus(self):
        self.luminari.connect_to_galactic_nexus("network1")
        self.assertTrue(self.luminari.galactic_nexus.is_connected("network1"))

    def test_disconnect_from_galactic_nexus(self):
        self.luminari.disconnect_from_galactic_nexus("network1")
        self.assertFalse(self.luminari.galactic_nexus.is_connected("network1"))

    def test_send_transaction(self):
        transaction = {"from": "0x123", "to": "0x456", "value": 1}
        self.luminari.send_transaction(transaction)

    def test_get_blockchain_state(self):
        state = self.luminari.get_blockchain_state()
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
