# dividend_contract.py

class DividendContract:
    def __init__(self):
        self.dividend_payments = {}

    def pay_dividend(self, amount):
        for user, balance in self.dividend_payments.items():
            user.transfer(amount * balance / self.total_supply)

    def get_dividend_payment(self, user):
        return self.dividend_payments.get(user, 0)
