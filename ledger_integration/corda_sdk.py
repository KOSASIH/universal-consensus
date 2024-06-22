# corda_sdk.py
import os
import json
from corda_rpc import CordaRPCClient

class CordaSDK:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        config_file = os.path.join(os.path.dirname(__file__), 'corda_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config

    def get_rpc_client(self):
        rpc_config = self.config['rpc']
        return CordaRPCClient(
            rpc_config['host'],
            rpc_config['port'],
            rpc_config['username'],
            rpc_config['password'],
            rpc_config['ssl']
        )

    def get_node_info(self):
        node_config = self.config['node']
        return {
            'node_info': node_config['node_info'],
            'node_config': node_config['node_config']
        }

    def get_flow(self, flow_name):
        flow_config = self.config['flows'][flow_name]
        return flow_config

# Example corda_config.json file
{
    "rpc": {
        "host": "localhost",
        "port": 10006,
        "username": "user1",
        "password": "password",
        "ssl": true
    },
    "node": {
        "node_info": {
            "node_name": "NodeA",
            "node_type": "CORDA_NODE"
        },
        "node_config": {
            "p2pAddress": "localhost:10007"
        }
    },
    "flows": {
        "flow1": {
            "flow_name": "Flow1",
            "flow_version": "1.0"
        },
        "flow2": {
            "flow_name": "Flow2",
            "flow_version": "1.0"
        }
    }
}
