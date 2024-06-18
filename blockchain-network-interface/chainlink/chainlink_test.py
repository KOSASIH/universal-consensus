import os
import json
import pytest
from chainlink import Chainlink

@pytest.fixture
def chainlink_config():
    with open('chainlink_config.json') as f:
        return json.load(f)

@pytest.fixture
def chainlink_instance(chainlink_config):
    return Chainlink(chainlink_config)

def test_get_price(chainlink_instance):
    asset = 'ETH'
    price = chainlink_instance.get_price(asset)
    assert price is not None
    assert price > 0

def test_get_conversion_rate(chainlink_instance):
    from_asset = 'ETH'
    to_asset = 'BTC'
    rate = chainlink_instance.get_conversion_rate(from_asset, to_asset)
    assert rate is not None
    assert rate > 0

def test_request_data(chainlink_instance):
    asset = 'ETH'
    callback_address = '0x...YOUR_CALLBACK_ADDRESS...'
    request_id = chainlink_instance.request_data(asset, callback_address)
    assert request_id is not None

def test_fulfill_data(chainlink_instance):
    request_id = '0x...YOUR_REQUEST_ID...'
    data = '0x...YOUR_DATA...'
    result = chainlink_instance.fulfill_data(request_id, data)
    assert result is not None

def test_get_request_status(chainlink_instance):
    request_id = '0x...YOUR_REQUEST_ID...'
    status = chainlink_instance.get_request_status(request_id)
    assert status is not None
    assert status in ['pending', 'fulfilled', 'errored']
