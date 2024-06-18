import unittest
from corda import Corda

class TestCorda(unittest.TestCase):
    def setUp(self):
        self.corda = Corda()

    def test_load_config(self):
        self.assertIsNotNone(self.corda.config)

    def test_get_config(self):
        self.assertEqual(self.corda.get_config('node_url'), 'http://localhost:10000')

    def test_create_transaction(self):
        state = {'data': 'Hello, Corda!'}
        contract = 'com.example.Contract'
        notary = 'O=Notary, L=London, C=GB'
        tx = self.corda.create_transaction(state, contract, notary)
        self.assertIsNotNone(tx)

    def test_sign_transaction(self):
        state = {'data': 'Hello, Corda!'}
        contract = 'com.example.Contract'
        notary = 'O=Notary, L=London, C=GB'
        tx = self.corda.create_transaction(state, contract, notary)
        signed_tx = self.corda.sign_transaction(tx)
        self.assertIsNotNone(signed_tx)

    def test_submit_transaction(self):
        state = {'data': 'Hello, Corda!'}
        contract = 'com.example.Contract'
        notary = 'O=Notary, L=London, C=GB'
        tx = self.corda.create_transaction(state, contract, notary)
        signed_tx = self.corda.sign_transaction(tx)
        self.corda.submit_transaction(signed_tx)

    def test_query_state(self):
        state_ref = '1234567890'
        state = self.corda.query_state(state_ref)
        self.assertIsNotNone(state)

if __name__ == '__main__':
    unittest.main()
