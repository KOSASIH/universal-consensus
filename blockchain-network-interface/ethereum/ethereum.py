# blockchain-network-interface/ethereum/ethereum.py
import json
import requests

class Ethereum:
    def __init__(self, node_url: str):
        self.node_url = node_url

    def get_block_by_number(self, block_number: int) -> dict:
        response = requests.get(f"{self.node_url}/block/{block_number}")
        return json.loads(response.content)

    def send_transaction(self, tx: dict) -> str:
        response = requests.post(f"{self.node_url}/transaction", json=tx)
        return response.content.decode("utf-8")
