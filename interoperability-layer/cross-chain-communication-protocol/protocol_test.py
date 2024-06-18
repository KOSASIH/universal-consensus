# cross-chain-communication-protocol/protocol_test.py
import unittest
from protocol import CrossChainCommunicationProtocol

class TestCrossChainCommunicationProtocol(unittest.TestCase):
    def setUp(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.chain_id = "test_chain"
        self.protocol = CrossChainCommunicationProtocol(self.private_key, self.chain_id)

    def test_generate_nonce(self):
        nonce = self.protocol.generate_nonce()
        self.assertEqual(len(nonce), 64)

    def test_create_message(self):
        payload = {"key": "value"}
        nonce = self.protocol.generate_nonce()
        message = self.protocol.create_message(payload, nonce)
        self.assertIn("payload", message)
        self.assertIn("nonce", message)
        self.assertIn("chain_id", message)

    def test_sign_message(self):
        message = "Hello, world!"
        signature = self.protocol.sign_message(message)
        self.assertEqual(len(signature), 64)

    def test_verify_signature(self):
        message = "Hello, world!"
        signature = self.protocol.sign_message(message)
        self.assertTrue(self.protocol.verify_signature(message, signature))

    def test_derive_shared_secret(self):
        public_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        ).public_key()
        shared_secret = self.protocol.derive_shared_secret(public_key)
        self.assertEqual(len(shared_secret), 32)

if __name__ == "__main__":
    unittest.main()
