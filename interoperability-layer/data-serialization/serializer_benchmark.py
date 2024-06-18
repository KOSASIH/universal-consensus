# interoperability-layer/data-serialization/serializer_benchmark.py
import timeit
from serializer import Serializer

def benchmark_json_serialization():
    serializer = Serializer("json")
    data = {"key": "value"}
    serialized_data = serializer.serialize(data)
    deserialized_data = serializer.deserialize(serialized_data)
    return timeit.timeit(lambda: serializer.serialize(data), number=1000)

def benchmark_msgpack_serialization():
    serializer = Serializer("msgpack")
    data = {"key": "value"}
    serialized_data = serializer.serialize(data)
    deserialized_data = serializer.deserialize(serialized_data)
    return timeit.timeit(lambda: serializer.serialize(data), number=1000)

def benchmark_avro_serialization():
    serializer = Serializer("avro")
    data = {"key": "value"}
    serialized_data = serializer.serialize(data)
    deserialized_data = serializer.deserialize(serialized_data)
    return timeit.timeit(lambda: serializer.serialize(data), number=1000)

print("JSON serialization benchmark:", benchmark_json_serialization())
print("MsgPack serialization benchmark:", benchmark_msgpack_serialization())
print("Avro serialization benchmark:", benchmark_avro_serialization())
