import os
import hashlib
import pickle
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from cryptography.fernet import Fernet
from grpc import server, rpc
from google.protobuf.json_format import MessageToDict

from.storage_pb2 import StorageRequest, StorageResponse
from.storage_pb2_grpc import StorageServicer

class DistributedStorage:
    def __init__(self, nodes: List[str], encryption_key: str):
        self.nodes = nodes
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key.encode())
        self.executor = ThreadPoolExecutor(max_workers=10)

    def put(self, key: str, value: Any) -> None:
        """Store a value in the distributed storage"""
        pickled_value = pickle.dumps(value)
        encrypted_value = self.fernet.encrypt(pickled_value)
        hash_key = hashlib.sha256(key.encode()).hexdigest()
        node = self.get_node(hash_key)
        self.executor.submit(self._put, node, hash_key, encrypted_value)

    def get(self, key: str) -> Any:
        """Retrieve a value from the distributed storage"""
        hash_key = hashlib.sha256(key.encode()).hexdigest()
        node = self.get_node(hash_key)
        response = self._get(node, hash_key)
        if response:
            encrypted_value = response.value
            pickled_value = self.fernet.decrypt(encrypted_value)
            return pickle.loads(pickled_value)
        return None

    def _put(self, node: str, key: str, value: bytes) -> None:
        """Internal put method, sends a gRPC request to the node"""
        channel = grpc.insecure_channel(node)
        stub = StorageServicer(channel)
        request = StorageRequest(key=key, value=value)
        response = stub.put(request)
        if response.status!= 200:
            raise Exception(f"Error storing value: {response.message}")

    def _get(self, node: str, key: str) -> StorageResponse:
        """Internal get method, sends a gRPC request to the node"""
        channel = grpc.insecure_channel(node)
        stub = StorageServicer(channel)
        request = StorageRequest(key=key)
        response = stub.get(request)
        return response

    def get_node(self, hash_key: str) -> str:
        """Determine the node responsible for storing the value"""
        node_index = int(hash_key, 16) % len(self.nodes)
        return self.nodes[node_index]

def create_storage_servicer(encryption_key: str) -> StorageServicer:
    """Create a gRPC servicer for the storage node"""
    storage = DistributedStorage([], encryption_key)
    return StorageServicer(storage)

def serve(storage_node: str, encryption_key: str) -> None:
    """Start the gRPC server for the storage node"""
    servicer = create_storage_servicer(encryption_key)
    server.add_to_server(servicer, storage_node)
    print(f"Storage node listening on {storage_node}")
    server.serve_forever()
