import hashlib
import hmac
import os
import secrets
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class Wallet:
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.address = None

    def generate_keys(self):
        private_key = ec.generate_private_key(ec.SECP256k1(), default_backend())
        public_key = private_key.public_key()
        self.private_key = private_key
        self.public_key = public_key
        self.address = self.generate_address()

    def generate_address(self):
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        address = hashlib.sha256(public_key_bytes).hexdigest()
        return address

    def sign_transaction(self, transaction):
        signature = self.private_key.sign(
            transaction.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )
        return signature.hex()

    def verify_transaction(self, transaction, signature):
        try:
            self.public_key.verify(
                bytes.fromhex(signature),
                transaction.encode('utf-8'),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except:
            return False

    def get_balance(self, sidra_chain):
        return sidra_chain.get_balance(self.address)

    def send_transaction(self, sidra_chain, to_address, amount):
        transaction = {
            'from': self.address,
            'to': to_address,
            'amount': amount
        }
        signature = self.sign_transaction(str(transaction))
        transaction['signature'] = signature
        sidra_chain.add_transaction(transaction)
        return transaction
