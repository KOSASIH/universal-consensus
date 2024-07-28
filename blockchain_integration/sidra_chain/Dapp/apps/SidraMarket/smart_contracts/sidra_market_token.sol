
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract SidraMarketToken {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function mint(address _to, uint256 _amount) public {
        require(msg.sender == owner, "Only the owner can mint tokens");
        balances[_to] += _amount;
    }

    function transfer(address _from, address _to, uint256 _amount) public {
        require(balances[_from] >= _amount, "Insufficient balance");
        balances[_from] -= _amount;
        balances[_to] += _amount;
    }
}
