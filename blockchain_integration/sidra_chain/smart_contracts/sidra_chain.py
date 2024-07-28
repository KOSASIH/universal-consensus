import hashlib
import hmac
import os
import secrets
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class SidraChain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 2
        self.miner_rewards = 10
        self.block_time = 60

    def create_genesis_block(self):
        genesis_block = Block(0, "Genesis Block", "0", int(time.time()), [])
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self, miner):
        if not self.pending_transactions:
            return False

        latest_block = self.get_latest_block()
        new_block_index = latest_block.index + 1
        new_block_timestamp = int(time.time())
        new_block_data = self.pending_transactions[:]
        new_block_hash = self.calculate_hash(new_block_index, latest_block.hash, new_block_timestamp, new_block_data)
        new_block = Block(new_block_index, new_block_hash, latest_block.hash, new_block_timestamp, new_block_data)

        self.pending_transactions = []
        self.chain.append(new_block)
        return new_block

    def calculate_hash(self, index, previous_hash, timestamp, data):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != self.calculate_hash(current_block.index, previous_block.hash, current_block.timestamp, current_block.data):
                return False
        return True

    def get_balance(self, address):
        balance = 0
        for block in self.chain:
            for transaction in block.data:
                if transaction['from'] == address:
                    balance -= transaction['amount']
                elif transaction['to'] == address:
                    balance += transaction['amount']
        return balance

    def create_transaction(self, from_address, to_address, amount):
        transaction = {
            'from': from_address,
            'to': to_address,
            'amount': amount
        }
        self.add_transaction(transaction)
        return transaction

    def get_transaction(self, transaction_hash):
        for block in self.chain:
            for transaction in block.data:
                if transaction['hash'] == transaction_hash:
                    return transaction
        return None

class Block:
    def __init__(self, index, hash, previous_hash, timestamp, data):
        self.index = index
        self.hash = hash
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data

class Transaction:
    def __init__(self, from_address, to_address, amount):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.hash = hashlib.sha256((str(from_address) + str(to_address) + str(amount)).encode('utf-8')).hexdigest()
