// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReputationSystem {
    // Mapping of users to their respective reputations
    mapping (address => uint256) public reputations;

    // Event emitted when a user's reputation is updated
    event ReputationUpdated(address indexed user, uint256 reputation);

    // Function to update a user's reputation
    function updateReputation(address _user, uint256 _reputation) public {
        // Set the user's reputation
        reputations[_user] = _reputation;

        // Emit the ReputationUpdated event
        emit ReputationUpdated(_user, _reputation);
    }
}
