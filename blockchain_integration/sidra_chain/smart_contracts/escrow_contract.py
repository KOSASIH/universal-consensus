# escrow_contract.py

class EscrowContract:
    def __init__(self):
        self.escrowed_coins = {}

    def escrow(self, amount, condition):
        self.escrowed_coins[msg.sender] = (amount, condition)

    def release(self):
        if self.escrowed_coins[msg.sender][1] == True:
            msg.sender.transfer(self.escrowed_coins[msg.sender][0])
            del self.escrowed_coins[msg.sender]
