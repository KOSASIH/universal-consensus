import time
from distributed_storage_layer.storage import DistributedStorage

def benchmark_put_get(storage: DistributedStorage, num_iterations: int) -> None:
    key = "my_key"
    value = {"foo": "bar"}
    start_time = time.time()
    for _ in range(num_iterations):
        storage.put(key, value)
        storage.get(key)
    end_time = time.time()
    print(f"Average put-get time: {(end_time - start_time) / num_iterations:.2f}ms")

if __name__ == "__main__":
    nodes = ["node1:50051", "node2:50052", "node3:50053"]
    encryption_key = "secret_key"
    storage = DistributedStorage(nodes, encryption_key)
    benchmark_put_get(storage, 1000)
