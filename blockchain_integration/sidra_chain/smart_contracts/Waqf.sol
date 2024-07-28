pragma solidity ^0.8.0;

contract Waqf is Owner {
    mapping(address => uint256) public balances;

    event Transfer(address indexed from, address indexed to, uint256 value);

    function transfer(address to, uint256 value) public onlyOwner {
        require(balances[msg.sender] >= value, "Waqf: insufficient balance");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    function getBalance(address account) public view returns (uint256) {
        return balances[account];
    }
}
