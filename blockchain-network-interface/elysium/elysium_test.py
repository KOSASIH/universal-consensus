import unittest
from elysium import Elysium

class TestElysium(unittest.TestCase):
    def setUp(self):
        self.elysium = Elysium()

    def test_load_config(self):
        self.assertIsNotNone(self.elysium.config)

    def test_create_network(self):
        self.assertIsNotNone(self.elysium.network)

    def test_start_network(self):
        self.elysium.start_network()
        self.assertTrue(self.elysium.network.is_running())
        self.assertTrue(self.elysium.galactic_nexus.is_running())

    def test_stop_network(self):
        self.elysium.stop_network()
        self.assertFalse(self.elysium.network.is_running())
        self.assertFalse(self.elysium.galactic_nexus.is_running())

    def test_connect_to_galactic_nexus(self):
        self.elysium.connect_to_galactic_nexus("network1")
        self.assertTrue(self.elysium.galactic_nexus.is_connected("network1"))

    def test_disconnect_from_galactic_nexus(self):
        self.elysium.disconnect_from_galactic_nexus("network1")
        self.assertFalse(self.elysium.galactic_nexus.is_connected("network1"))

if __name__ == '__main__':
    unittest.main()
