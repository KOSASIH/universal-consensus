import argparse
import getpass
import hashlib
import hmac
import os
import secrets
import time
from wallet import Wallet
from node import Node

def main():
    parser = argparse.ArgumentParser(description='Sidra Chain CLI')
    parser.add_argument('--wallet', action='store_true', help='Create a new wallet')
    parser.add_argument('--node', action='store_true', help='Start a new node')
    parser.add_argument('--send', action='store_true', help='Send a transaction')
    parser.add_argument('--get-balance', action='store_true', help='Get the balance of an address')
    parser.add_argument('--get-blockchain', action='store_true', help='Get the blockchain')
    args = parser.parse_args()

    if args.wallet:
        wallet = Wallet()
        wallet.generate_keys()
        print('Private key:', wallet.private_key)
        print('Public key:', wallet.public_key)
        print('Address:', wallet.address)

    elif args.node:
        node = Node()
        node.start()

    elif args.send:
        wallet = Wallet()
        to_address = input('Enter the recipient address: ')
                amount = int(input('Enter the amount: '))
        wallet.send_transaction(node.sidra_chain, to_address, amount)
        print('Transaction sent!')

    elif args.get_balance:
        wallet = Wallet()
        address = input('Enter the address: ')
        balance = node.sidra_chain.get_balance(address)
        print('Balance:', balance)

    elif args.get_blockchain:
        blockchain = node.sidra_chain.chain
        print('Blockchain:')
        for block in blockchain:
            print(block)

if __name__ == '__main__':
    main()
