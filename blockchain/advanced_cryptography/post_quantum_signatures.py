import hashlib
from sphincs import SPHINCS

class PostQuantumSignatures:
    def __init__(self, private_key):
        self.private_key = private_key
        self.sphincs = SPHINCS()

    def sign(self, message):
        # Sign the message using SPHINCS
        signature = self.sphincs.sign(self.private_key, message)
        return signature

    def verify(self, message, signature):
        # Verify the signature using SPHINCS
        return self.sphincs.verify(self.private_key, message, signature)
