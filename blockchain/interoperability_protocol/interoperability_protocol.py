import hashlib
from cosmos_sdk import CosmosSDK

class InteroperabilityProtocol:
    def __init__(self, cosmos_sdk):
        self.cosmos_sdk = cosmos_sdk

    def create_channel(self, source_chain, destination_chain):
        # Create a channel between two chains
        channel_id = hashlib.sha256(str(source_chain + destination_chain).encode()).hexdigest()
        self.cosmos_sdk.create_channel(channel_id, source_chain, destination_chain)
        return channel_id

    def send_packet(self, channel_id, packet):
        # Send a packet through the channel
        self.cosmos_sdk.send_packet(channel_id, packet)

    def receive_packet(self, channel_id):
        # Receive a packet from the channel
        packet = self.cosmos_sdk.receive_packet(channel_id)
        return packet
