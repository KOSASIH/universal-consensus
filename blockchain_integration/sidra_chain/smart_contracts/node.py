import hashlib
import hmac
import os
import secrets
import time
import socket
import threading
from sidra_chain import SidraChain

class Node:
    def __init__(self):
        self.sidra_chain = SidraChain()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('localhost', 8080))
        self.socket.listen(5)
        self.peers = []

    def start(self):
        threading.Thread(target=self.listen_for_connections).start()
        threading.Thread(target=self.mine_block).start()

    def listen_for_connections(self):
        while True:
            connection, address = self.socket.accept()
            threading.Thread(target=self.handle_connection, args=(connection, address)).start()

    def handle_connection(self, connection, address):
        while True:
            data = connection.recv(1024)
            if not data:
                break
            message = data.decode('utf-8')
            if message.startswith('get_blockchain'):
                self.send_blockchain(connection)
            elif message.startswith('add_transaction'):
                self.add_transaction(message.split(':')[1])
            elif message.startswith('mine_block'):
                self.mine_block()
            elif message.startswith('get_peers'):
                self.send_peers(connection)
            elif message.startswith('add_peer'):
                self.add_peer(message.split(':')[1])

    def send_blockchain(self, connection):
        blockchain = self.sidra_chain.chain
        connection.sendall(str(blockchain).encode('utf-8'))

    def add_transaction(self, transaction):
        self.sidra_chain.add_transaction(transaction)

    def mine_block(self):
        self.sidra_chain.mine_block()

    def send_peers(self, connection):
        peers = self.peers
        connection.sendall(str(peers).encode('utf-8'))

    def add_peer(self, peer):
        self.peers.append(peer)
