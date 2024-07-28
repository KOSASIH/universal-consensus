# voting_contract.py

class VotingContract:
    def __init__(self):
        self.proposals = {}
        self.votes = {}

    def add_proposal(self, proposal):
        self.proposals[proposal] = 0

    def vote(self, proposal):
        self.votes[msg.sender] = proposal
        self.proposals[proposal] += 1

    def get_winner(self):
        max_votes = 0
        winner = None
        for proposal, votes in self.proposals.items():
            if votes > max_votes:
                max_votes = votes
                winner = proposal
        return winner
