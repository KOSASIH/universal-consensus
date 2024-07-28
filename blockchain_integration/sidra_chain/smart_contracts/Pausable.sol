pragma solidity ^0.8.0;

contract Pausable {
    bool public paused;

    event Paused(address account);
    event Unpaused(address account);

    modifier onlyWhenNotPaused() {
        require(!paused, "Pausable: paused");
        _;
    }

    modifier onlyWhenPaused() {
        require(paused, "Pausable: not paused");
        _;
    }

    function pause() public onlyWhenNotPaused {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() public onlyWhenPaused {
        paused = false;
        emit Unpaused(msg.sender);
    }
}
