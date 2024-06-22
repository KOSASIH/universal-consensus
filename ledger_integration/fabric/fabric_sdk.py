# fabric_sdk.py
from fabric_sdk_py import FabricClient

class FabricLedger:
    def __init__(self, channel_name, chaincode_name):
        self.client = FabricClient()
        self.channel_name = channel_name
        self.chaincode_name = chaincode_name

    def invoke_chaincode(self, func, args):
        response = self.client.invoke_chaincode(self.channel_name, self.chaincode_name, func, args)
        return response
