import unittest
from unittest.mock import patch, MagicMock
from distributed_storage_layer.storage import DistributedStorage

class TestDistributedStorage(unittest.TestCase):
    def setUp(self):
        self.nodes = ["node1:50051", "node2:50052", "node3:50053"]
        self.encryption_key = "secret_key"
        self.storage = DistributedStorage(self.nodes, self.encryption_key)

    def test_put_get(self):
        key = "my_key"
        value = {"foo": "bar"}
        self.storage.put(key, value)
        retrieved_value = self.storage.get(key)
        self.assertEqual(value, retrieved_value)

    def test_put_get_with_encryption(self):
        key = "my_key"
        value = {"foo": "bar"}
        self.storage.put(key, value)
        encrypted_value = self.storage._get(self.nodes[0], key).value
        decrypted_value = self.storage.fernet.decrypt(encrypted_value)
        self.assertEqual(value, pickle.loads(decrypted_value))

    @patch("distributed_storage_layer.storage.grpc")
    def test_serve(self, mock_grpc):
        mock_grpc.server.return_value = MagicMock()
        serve("node1:50051", self.encryption_key)
        mock_grpc.server.assert_called_once_with("node1:50051")

if __name__ == "__main__":
    unittest.main()
