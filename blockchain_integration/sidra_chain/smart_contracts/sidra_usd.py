import smart_contract
import blockchain
import cryptography

class SidraUSD(smart_contract.SmartContract):
    def __init__(self):
        super().__init__()
        self.name = "Sidra USD"
        self.symbol = "SUSD"
        self.decimals = 18
        self.total_supply = 100000000
        self.balances = {}

    def transfer(self, sender, recipient, amount):
        if sender not in self.balances:
            raise ValueError("Sender does not have a balance")
        if recipient not in self.balances:
            self.balances[recipient] = 0
        if amount > self.balances[sender]:
            raise ValueError("Insufficient balance")
        self.balances[sender] -= amount
        self.balances[recipient] += amount
        blockchain.broadcast_transaction(self, sender, recipient, amount)

    def balance_of(self, account):
        return self.balances.get(account, 0)

    def mint(self, amount):
        if amount > self.total_supply:
            raise ValueError("Cannot mint more than total supply")
        self.total_supply -= amount
        self.balances[self.owner] += amount

    def burn(self, amount):
        if amount > self.balances[self.owner]:
            raise ValueError("Insufficient balance")
        self.balances[self.owner] -= amount
        self.total_supply += amount

    def get_owner(self):
        return self.owner

    def set_owner(self, new_owner):
        self.owner = new_owner

    def get_name(self):
        return self.name

    def get_symbol(self):
        return self.symbol

    def get_decimals(self):
        return self.decimals

    def get_total_supply(self):
        return self.total_supply

    def get_balances(self):
        return self.balances

    def verify_transaction(self, transaction):
        sender = transaction["sender"]
        recipient = transaction["recipient"]
        amount = transaction["amount"]
        if sender not in self.balances:
            raise ValueError("Sender does not have a balance")
        if recipient not in self.balances:
            self.balances[recipient] = 0
        if amount > self.balances[sender]:
            raise ValueError("Insufficient balance")
        return True

    def sign_transaction(self, transaction):
        private_key = self.owner_private_key
        signature = cryptography.sign(transaction, private_key)
        return signature

    def verify_signature(self, transaction, signature):
        public_key = self.owner_public_key
        return cryptography.verify(transaction, signature, public_key)
