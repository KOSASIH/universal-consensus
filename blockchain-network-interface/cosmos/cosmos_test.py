import unittest
from cosmos import Cosmos

class TestCosmos(unittest.TestCase):
    def setUp(self):
        self.cosmos = Cosmos()

    def test_load_config(self):
        self.assertIsNotNone(self.cosmos.config)

    def test_get_config(self):
        self.assertEqual(self.cosmos.get_config('chain_id'), 'cosmos-hub-4')

    def test_get_config_default(self):
        self.assertIsNone(self.cosmos.get_config('non_existent_key'))

if __name__ == '__main__':
    unittest.main()
