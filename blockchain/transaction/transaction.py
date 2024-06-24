# Transaction implementation
import hashlib

class Transaction:
    def __init__(self, sender: str, recipient: str, amount: int):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        # Calculate transaction hash implementation
        pass
