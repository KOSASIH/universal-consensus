# interoperability-layer/data-serialization/serializer.py
import json
import msgpack
import avro.schema
import avro.io

class Serializer:
    def __init__(self, format: str):
        self.format = format

    def serialize(self, data: dict) -> bytes:
        if self.format == "json":
            return json.dumps(data).encode("utf-8")
        elif self.format == "msgpack":
            return msgpack.packb(data)
        elif self.format == "avro":
            schema = avro.schema.parse("""
                {
                    "type": "record",
                    "name": "Data",
                    "fields": [
                        {"name": "key", "type": "string"},
                        {"name": "value", "type": "string"}
                    ]
                }
            """)
            writer = avro.io.DatumWriter(schema)
            bytes_writer = avro.io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            writer.write(data, encoder)
            return bytes_writer.getvalue()
        else:
            raise ValueError("Unsupported serialization format")

    def deserialize(self, data: bytes) -> dict:
        if self.format == "json":
            return json.loads(data.decode("utf-8"))
        elif self.format == "msgpack":
            return msgpack.unpackb(data)
        elif self.format == "avro":
            schema = avro.schema.parse("""
                {
                    "type": "record",
                    "name": "Data",
                    "fields": [
                        {"name": "key", "type": "string"},
                        {"name": "value", "type": "string"}
                    ]
                }
            """)
            bytes_reader = avro.io.BytesIO(data)
            decoder = avro.io.BinaryDecoder(bytes_reader)
            reader = avro.io.DatumReader(schema)
            return reader.read(decoder)
        else:
            raise ValueError("Unsupported serialization format")
