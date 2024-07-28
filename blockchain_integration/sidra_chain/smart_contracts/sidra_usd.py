# -*- coding: utf-8 -*-

# Import necessary libraries
from solidity import contract, interface
from solidity.types import uint256, address, bytes32
from solidity.functions import pure, view, payable

# Define the Sidra USD contract
@contract
class SidraUSD:
    # Define the contract variables
    _name: bytes32 = 'Sidra USD'
    _symbol: bytes32 = 'SDUSD'
    _decimals: uint256 = 18
    _total_supply: uint256 = 100000000 * (10 ** _decimals)
    _owner: address

    # Define the contract events
    Transfer: event({_from: address, _to: address, _value: uint256})
    Approval: event({_owner: address, _spender: address, _value: uint256})

    # Define the contract constructor
    def __init__(_owner: address):
        self._owner = _owner

    # Define the transfer function
    @payable
    def transfer(_to: address, _value: uint256) -> bool:
        if _value > 0 and _to != address(0):
            if self.balanceOf(msg.sender) >= _value:
                self.balanceOf[msg.sender] -= _value
                self.balanceOf[_to] += _value
                emit Transfer(msg.sender, _to, _value)
                return True
        return False

    # Define the approve function
    def approve(_spender: address, _value: uint256) -> bool:
        if _spender != address(0):
            self.allowance[msg.sender][_spender] = _value
            emit Approval(msg.sender, _spender, _value)
            return True
        return False

    # Define the transferFrom function
    @payable
    def transferFrom(_from: address, _to: address, _value: uint256) -> bool:
        if _value > 0 and _to != address(0) and _from != address(0):
            if self.balanceOf(_from) >= _value and self.allowance[_from][msg.sender] >= _value:
                self.balanceOf[_from] -= _value
                self.balanceOf[_to] += _value
                self.allowance[_from][msg.sender] -= _value
                emit Transfer(_from, _to, _value)
                return True
        return False

    # Define the balanceOf function
    @view
    def balanceOf(_owner: address) -> uint256:
        return self.balanceOf[_owner]

    # Define the allowance function
    @view
    def allowance(_owner: address, _spender: address) -> uint256:
        return self.allowance[_owner][_spender]

    # Define the totalSupply function
    @view
    def totalSupply() -> uint256:
        return self._total_supply
