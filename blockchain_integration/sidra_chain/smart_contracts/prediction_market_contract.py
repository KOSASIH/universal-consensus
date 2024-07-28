# prediction_market_contract.py

class PredictionMarketContract:
    def __init__(self):
        self.predictions = {}
        self.outcomes = {}

    def predict(self, event, outcome):
        self.predictions[msg.sender] = (event, outcome)

    def resolve(self, event, outcome):
        for user, prediction in self.predictions.items():
            if prediction[0] == event and prediction[1] == outcome:
                user.transfer(self.reward_amount)
