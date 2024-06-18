import unittest
import asyncio
import json
import os
from galactic_consensus_nexus import GalacticConsensusNexus

class TestGalacticConsensusNexus(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config_path = 'galactic_consensus_nexus_config.json'
        self.nexus = GalacticConsensusNexus(self.config_path)

    async def test_node_registration(self):
        node_id = 'GCN-002'
        node_addr = 'tcp://0.0.0.0:8081'
        await self.nexus.register_node(node_id, node_addr)
        self.assertIn(node_id, self.nexus.node_manager.nodes)

    async def test_node_discovery(self):
        await self.nexus.discover_nodes()
        self.assertGreater(len(self.nexus.node_manager.nodes), 1)

    async def test_consensus_algorithm(self):
        await self.nexus.consensus_engine.run_consensus()
        self.assertTrue(self.nexus.consensus_engine.consensus_reached)

    async def test_transaction_validation(self):
        tx = {'from': 'GCN-001', 'to': 'GCN-002', 'amount': 10}
        await self.nexus.tx_manager.validate_transaction(tx)
        self.assertTrue(self.nexus.tx_manager.tx_pool)

    async def test_transaction_propagation(self):
        tx = {'from': 'GCN-001', 'to': 'GCN-002', 'amount': 10}
        await self.nexus.tx_manager.propagate_transaction(tx)
        self.assertTrue(self.nexus.tx_manager.tx_pool)

    async def test_blockchain_storage(self):
        block = {'block_number': 1, 'transactions': []}
        await self.nexus.blockchain_store.add_block(block)
        self.assertTrue(self.nexus.blockchain_store.blockchain)

    async def test_network_communication(self):
        message = {'type': 'hello', 'data': 'Hello, Galactic Consensus Nexus!'}
        await self.nexus.network_communicator.send_message(message, 'tcp://0.0.0.0:8081')
        self.assertTrue(self.nexus.network_communicator.sent_messages)

    async def test_config_loading(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.assertEqual(self.nexus.config, config)

    async def test_logging(self):
        log_message = 'Test log message'
        self.nexus.logger.debug(log_message)
        self.assertIn(log_message, self.nexus.logger.log_messages)

    async def test_metrics(self):
        metrics = self.nexus.metrics
        self.assertTrue(metrics.enabled)
        self.assertEqual(metrics.port, 8081)

if __name__ == '__main__':
    unittest.main()
