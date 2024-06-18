import json
import os

class Cosmos:
    def __init__(self, config_file='cosmos_config.json'):
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return config

    def get_config(self, key=None):
        if key is None:
            return self.config
        else:
            return self.config.get(key)

if __name__ == '__main__':
    cosmos = Cosmos()
    print(cosmos.get_config('chain_id'))
