pragma solidity ^0.8.0;

contract SidraToken is Pausable, Owner {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    uint256 public circulatingSupply;

    mapping(address => uint256) public balances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function transfer(address to, uint256 value) public onlyWhenNotPaused {
        require(balances[msg.sender] >= value, "SidraToken: insufficient balance");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    function approve(address spender, uint256 value) public onlyWhenNotPaused {
        require(balances[msg.sender] >= value, "SidraToken: insufficient balance");
        allowed[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
    }

    function mint(address to, uint256 value) public onlyOwner {
        totalSupply += value;
        circulatingSupply += value;
        balances[to] += value;
        emit Transfer(address(0), to, value);
    }

    function pause() public onlyOwner {
        super.pause();
    }

    function unpause() public onlyOwner {
        super.unpause();
    }
}
