// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DisputeResolution {
    // Mapping of disputes to their respective details
    mapping (uint256 => Dispute) public disputes;

    // Struct to represent a dispute
    struct Dispute {
        uint256 productId;
        address buyer;
        address seller;
        string reason;
        string resolution;
        bool exists;
    }

    // Event emitted when a dispute is created
    event DisputeCreated(uint256 indexed productId, address indexed buyer, address indexed seller, string reason);

    // Event emitted when a dispute is resolved
    event DisputeResolved(uint256 indexed productId, address indexed buyer, address indexed seller, string resolution);

        // Function to create a dispute
    function createDispute(uint256 _productId, address _buyer, address _seller, string memory _reason) public {
        // Create a new dispute
        disputes[_productId] = Dispute(_productId, _buyer, _seller, _reason, "", true);

        // Emit the DisputeCreated event
        emit DisputeCreated(_productId, _buyer, _seller, _reason);
    }

    // Function to resolve a dispute
    function resolveDispute(uint256 _productId, string memory _resolution) public {
        // Check if the dispute exists
        require(disputes[_productId].exists, "Dispute does not exist");

        // Resolve the dispute
        disputes[_productId].resolution = _resolution;
        disputes[_productId].exists = false;

        // Emit the DisputeResolved event
        emit DisputeResolved(_productId, disputes[_productId].buyer, disputes[_productId].seller, _resolution);
    }
}
