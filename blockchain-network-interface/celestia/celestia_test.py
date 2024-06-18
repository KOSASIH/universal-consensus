import unittest
from celestia import Celestia

class TestCelestia(unittest.TestCase):
    def setUp(self):
        self.celestia = Celestia()

    def test_load_config(self):
        self.assertIsNotNone(self.celestia.config)

    def test_create_network(self):
        self.assertIsNotNone(self.celestia.network)

    def test_start_network(self):
        self.celestia.start_network()
        self.assertTrue(self.celestia.network.is_running())

    def test_stop_network(self):
        self.celestia.stop_network()
        self.assertFalse(self.celestia.network.is_running())

if __name__ == '__main__':
    unittest.main()
