import unittest
from unittest.mock import patch, MagicMock
from consensus_algorithms.pow import PoW
from consensus_algorithms.pos import PoS
from consensus_algorithms.dpos import DPoS
from consensus_algorithms.pbft import PBFT

class TestConsensusAlgorithms(unittest.TestCase):
    def setUp(self):
        self.pow = PoW()
        self.pos = PoS()
        self.dpos = DPoS()
        self.pbft = PBFT()

    def test_pow_mining(self):
        block = {"transactions": [{"from": "Alice", "to": "Bob", "amount": 10}]}
        self.pow.mine(block)
        self.assertEqual(self.pow.chain[-1], block)

    def test_pos_voting(self):
        validators = [{"id": "Alice", "stake": 100}, {"id": "Bob", "stake": 50}]
        block = {"transactions": [{"from": "Alice", "to": "Bob", "amount": 10}]}
        self.pos.vote(validators, block)
        self.assertEqual(self.pos.chain[-1], block)

    def test_dpos_delegation(self):
        delegators = [{"id": "Alice", "stake": 100}, {"id": "Bob", "stake": 50}]
        validators = [{"id": "Charlie", "stake": 200}]
        block = {"transactions": [{"from": "Alice", "to": "Bob", "amount": 10}]}
        self.dpos.delegate(delegators, validators, block)
        self.assertEqual(self.dpos.chain[-1], block)

    def test_pbft_consensus(self):
        nodes = [{"id": "Alice", "private_key": "alice_private_key"}, {"id": "Bob", "private_key": "bob_private_key"}]
        block = {"transactions": [{"from": "Alice", "to": "Bob", "amount": 10}]}
        self.pbft.consensus(nodes, block)
        self.assertEqual(self.pbft.chain[-1], block)

    @patch("consensus_algorithms.pow.PoW.mine")
    def test_pow_mining_with_mock(self, mock_mine):
        block = {"transactions": [{"from": "Alice", "to": "Bob", "amount": 10}]}
        self.pow.mine(block)
        mock_mine.assert_called_once_with(block)

if __name__ == "__main__":
    unittest.main()
