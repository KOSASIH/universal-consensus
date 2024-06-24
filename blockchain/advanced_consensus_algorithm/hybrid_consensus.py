import hashlib
import time
from collections import defaultdict

class HybridConsensus:
    def __init__(self, nodes, threshold):
        self.nodes = nodes
        self.threshold = threshold
        self.blockchain = []
        self.pending_transactions = []

    def verify_transaction(self, transaction):
        # BFT-style verification
        signatures = []
        for node in self.nodes:
            signature = node.sign_transaction(transaction)
            signatures.append(signature)
        if len(signatures) >= self.threshold:
            return True
        return False

    def create_block(self, transactions):
        # DPoS-style block creation
        block = {
            'transactions': transactions,
            'timestamp': int(time.time()),
            'hash': hashlib.sha256(str(transactions).encode()).hexdigest(),
            'previous_hash': self.blockchain[-1]['hash'] if self.blockchain else None
        }
        self.blockchain.append(block)
        return block

    def add_transaction(self, transaction):
        if self.verify_transaction(transaction):
            self.pending_transactions.append(transaction)
            if len(self.pending_transactions) >= self.threshold:
                self.create_block(self.pending_transactions)
                self.pending_transactions = []
