# corda_config.py
import os
import json

class CordaConfig:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        config_file = os.path.join(os.path.dirname(__file__), 'corda_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config

    def get_node_config(self):
        return self.config['node']

    def get_rpc_config(self):
        return self.config['rpc']

    def get_flow_config(self, flow_name):
        return self.config['flows'][flow_name]

# Example corda_config.json file (same as above)
