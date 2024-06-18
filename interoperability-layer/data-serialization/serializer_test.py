# interoperability-layer/data-serialization/serializer_test.py
import unittest
from serializer import Serializer

class TestSerializer(unittest.TestCase):
    def test_json_serialization(self):
        serializer = Serializer("json")
        data = {"key": "value"}
        serialized_data = serializer.serialize(data)
        self.assertEqual(serialized_data, b'{"key": "value"}')
        deserialized_data = serializer.deserialize(serialized_data)
        self.assertEqual(deserialized_data, data)

    def test_msgpack_serialization(self):
        serializer = Serializer("msgpack")
        data = {"key": "value"}
        serialized_data = serializer.serialize(data)
        self.assertIsNotNone(serialized_data)
        deserialized_data = serializer.deserialize(serialized_data)
        self.assertEqual(deserialized_data, data)

    def test_avro_serialization(self):
        serializer = Serializer("avro")
        data = {"key": "value"}
        serialized_data = serializer.serialize(data)
        self.assertIsNotNone(serialized_data)
        deserialized_data = serializer.deserialize(serialized_data)
        self.assertEqual(deserialized_data, data)

if __name__ == "__main__":
    unittest.main()
