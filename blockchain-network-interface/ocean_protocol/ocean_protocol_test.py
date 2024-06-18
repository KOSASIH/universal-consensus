import os
import json
import pytest
from ocean_protocol_sdk import Ocean
from ocean_protocol import OceanProtocol

@pytest.fixture
def ocean_protocol_config():
    with open('ocean_protocol_config.json') as f:
        return json.load(f)

@pytest.fixture
def ocean_protocol_instance(ocean_protocol_config):
    return OceanProtocol(
        provider_url=ocean_protocol_config['provider']['url'],
        account=ocean_protocol_config['accounts']['owner']['address'],
        private_key=ocean_protocol_config['accounts']['owner']['privateKey']
    )

def test_create_wallet(ocean_protocol_instance):
    wallet = ocean_protocol_instance.create_wallet()
    assert wallet is not None
    assert wallet['address'] is not None

def test_get_account_balance(ocean_protocol_instance):
    balance = ocean_protocol_instance.get_account_balance()
    assert balance is not None
    assert balance > 0

def test_create_datatoken(ocean_protocol_instance):
    metadata_uri = 'https://example.com/metadata.json'
    datatoken = ocean_protocol_instance.create_datatoken(metadata_uri)
    assert datatoken is not None
    assert datatoken['address'] is not None

def test_list_assets(ocean_protocol_instance):
    assets = ocean_protocol_instance.list_assets()
    assert assets is not None
    assert len(assets) > 0

def test_get_asset_details(ocean_protocol_instance):
    asset_id = 'dataset1'
    asset_details = ocean_protocol_instance.get_asset_details(asset_id)
    assert asset_details is not None
    assert asset_details['name'] == 'Dataset 1'

def test_create_order(ocean_protocol_instance):
    asset_id = 'dataset1'
    amount = '10.0'
    price = '15.0'
    order = ocean_protocol_instance.create_order(asset_id, amount, price)
    assert order is not None
    assert order['id'] is not None

def test_get_orders(ocean_protocol_instance):
    asset_id = 'dataset1'
    orders = ocean_protocol_instance.get_orders(asset_id)
    assert orders is not None
    assert len(orders) > 0

def test_execute_order(ocean_protocol_instance):
    order_id = '0x1234567890abcdef'
    result = ocean_protocol_instance.execute_order(order_id)
    assert result is not None
    assert result['status'] == 'uccess'

def test_get_transactions(ocean_protocol_instance):
    transactions = ocean_protocol_instance.get_transactions()
    assert transactions is not None
    assert len(transactions) > 0

def test_policies(ocean_protocol_instance):
    policy_id = 'accessControl'
    policy = ocean_protocol_instance.get_policy(policy_id)
    assert policy is not None
    assert policy['type'] == 'accessControl'

def test_access_control(ocean_protocol_instance):
    asset_id = 'dataset1'
    address = '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
    result = ocean_protocol_instance.check_access_control(asset_id, address)
    assert result is not None
    assert result['allowed'] is True
