import hashlib
import hmac
import os
import secrets

def generate_private_key():
    return secrets.token_bytes(32)

def generate_public_key(private_key):
    return hashlib.sha256(private_key).hexdigest()

def sign(message, private_key):
    return hmac.new(private_key, message.encode('utf-8'), hashlib.sha256).hexdigest()

def verify(message, signature, public_key):
    return hmac.compare_digest(signature, sign(message, public_key))

def encrypt(message, public_key):
    # Use a secure encryption algorithm like AES
    # For simplicity, we'll use a basic XOR cipher
    encrypted_message = bytearray()
    for i in range(len(message)):
        encrypted_message.append(message[i] ^ public_key[i % len(public_key)])
    return encrypted_message.hex()

def decrypt(encrypted_message, private_key):
    # Use a secure decryption algorithm like AES
    # For simplicity, we'll use a basic XOR cipher
    decrypted_message = bytearray()
    for i in range(len(encrypted_message)):
        decrypted_message.append(encrypted_message[i] ^ private_key[i % len(private_key)])
    return decrypted_message.decode('utf-8')

def generate_keypair():
    private_key = generate_private_key()
    public_key = generate_public_key(private_key)
    return private_key, public_key

def save_keypair(private_key, public_key, filename):
    with open(filename, 'wb') as f:
        f.write(private_key + b'\n' + public_key.encode('utf-8'))

def load_keypair(filename):
    with open(filename, 'rb') as f:
        private_key, public_key = f.read().split(b'\n')
    return private_key, public_key.decode('utf-8')
