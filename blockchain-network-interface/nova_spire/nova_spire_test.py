import unittest
from nova_spire import NovaSpire

class TestNovaSpire(unittest.TestCase):
    def setUp(self):
        self.nova_spire = NovaSpire()

    def test_load_config(self):
        self.assertIsNotNone(self.nova_spire.config)

    def test_create_network(self):
        self.assertIsNotNone(self.nova_spire.network)

    def test_create_nodes(self):
        self.assertIsNotNone(self.nova_spire.nodes)

    def test_start_network(self):
        self.nova_spire.start_network()
        self.assertTrue(self.nova_spire.network.is_running())
        self.assertTrue(self.nova_spire.galactic_nexus.is_running())
        for node in self.nova_spire.nodes:
            self.assertTrue(node.is_running())

    def test_stop_network(self):
        self.nova_spire.stop_network()
        self.assertFalse(self.nova_spire.network.is_running())
        self.assertFalse(self.nova_spire.galactic_nexus.is_running())
        for node in self.nova_spire.nodes:
            self.assertFalse(node.is_running())

    def test_connect_to_galactic_nexus(self):
        self.nova_spire.connect_to_galactic_nexus("network1")
        self.assertTrue(self.nova_spire.galactic_nexus.is_connected("network1"))

    def test_disconnect_from_galactic_nexus(self):
        self.nova_spire.disconnect_from_galactic_nexus("network1")
        self.assertFalse(self.nova_spire.galactic_nexus.is_connected("network1"))

    def test_send_transaction(self):
        transaction = {"from": "0x123", "to": "0x456", "value": 1}
        self.nova_spire.send_transaction(transaction)

    def test_get_network_state(self):
        state = self.nova_spire.get_network_state()
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
