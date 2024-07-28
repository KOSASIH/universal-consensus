# waqf_contract.py

class WaqfContract:
    def __init__(self):
        self.total_waqf = 0
        self.donors = {}

    def donate(self, amount):
        self.total_waqf += amount
        self.donors[msg.sender] = self.donors.get(msg.sender, 0) + amount

    def get_total_waqf(self):
        return self.total_waqf

    def get_donor_balance(self, donor):
        return self.donors.get(donor, 0)
