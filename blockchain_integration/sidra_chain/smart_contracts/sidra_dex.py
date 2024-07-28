# -*- coding: utf-8 -*-

# Import necessary libraries
from solidity import contract, interface
from solidity.types import uint256, address, bytes32
from solidity.functions import pure, view, payable

# Define the Sidra DEX contract
@contract
class SidraDEX:
    # Define the contract variables
    _owner: address
    _sidra_usd: address
    _token_pairs: mapping(address, address)
    _orders: mapping(address, mapping(uint256, Order))

    # Define the contract events
    NewOrder: event({_token: address, _amount: uint256, _price: uint256})
    CancelOrder: event({_token: address, _amount: uint256, _price: uint256})
    FillOrder: event({_token: address, _amount: uint256, _price: uint256, _buyer: address, _seller: address})

    # Define the contract constructor
    def __init__(_owner: address, _sidra_usd: address):
        self._owner = _owner
        self._sidra_usd = _sidra_usd

    # Define the createOrder function
    def createOrder(_token: address, _amount: uint256, _price: uint256) -> bool:
        if _token != address(0) and _amount > 0 and _price > 0:
            self._orders[_token][_amount] = Order(_amount, _price, msg.sender)
            emit NewOrder(_token, _amount, _price)
            return True

         # Define the cancelOrder function
    def cancelOrder(_token: address, _amount: uint256) -> bool:
        if _token != address(0) and _amount > 0:
            if self._orders[_token][_amount].exists:
                self._orders[_token][_amount].exists = False
                emit CancelOrder(_token, _amount, self._orders[_token][_amount].price)
                return True
        return False

    # Define the fillOrder function
    def fillOrder(_token: address, _amount: uint256, _price: uint256) -> bool:
        if _token != address(0) and _amount > 0 and _price > 0:
            if self._orders[_token][_amount].exists and self._orders[_token][_amount].price == _price:
                self._orders[_token][_amount].exists = False
                emit FillOrder(_token, _amount, _price, msg.sender, self._orders[_token][_amount].seller)
                return True
        return False

    # Define the getOrder function
    @view
    def getOrder(_token: address, _amount: uint256) -> Order:
        return self._orders[_token][_amount]

    # Define the getOrders function
    @view
    def getOrders(_token: address) -> mapping(uint256, Order):
        return self._orders[_token]

    # Define the tokenPairs function
    @view
    def tokenPairs() -> mapping(address, address):
        return self._token_pairs

    # Define the sidraUSD function
    @view
    def sidraUSD() -> address:
        return self._sidra_usd

# Define the Order struct
struct Order:
    amount: uint256
    price: uint256
    seller: address
    exists: bool
