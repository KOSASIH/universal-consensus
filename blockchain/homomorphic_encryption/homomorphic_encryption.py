import numpy as np
from fhe import FHE

class HomomorphicEncryption:
    def __init__(self, private_key):
        self.private_key = private_key
        self.fhe = FHE()

    def encrypt(self, message):
        # Encrypt the message using FHE
        ciphertext = self.fhe.encrypt(self.private_key, message)
        return ciphertext

    def evaluate(self, ciphertext, function):
        # Evaluate a function on the encrypted ciphertext
        result = self.fhe.evaluate(ciphertext, function)
        return result

    def decrypt(self, ciphertext):
        # Decrypt the ciphertext using FHE
        message = self.fhe.decrypt(self.private_key, ciphertext)
        return message
