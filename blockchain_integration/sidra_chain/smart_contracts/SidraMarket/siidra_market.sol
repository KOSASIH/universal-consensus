// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./ReputationSystem.sol";
import "./DisputeResolution.sol";

contract SidraMarket {
    // Mapping of products to their respective sellers
    mapping (uint256 => address) public productSellers;

    // Mapping of sellers to their respective products
    mapping (address => uint256[]) public sellerProducts;

    // Mapping of buyers to their respective purchases
    mapping (address => uint256[]) public buyerPurchases;

    // Mapping of products to their respective prices
    mapping (uint256 => uint256) public productPrices;

    // Mapping of products to their respective descriptions
    mapping (uint256 => string) public productDescriptions;

    // Mapping of products to their respective images
    mapping (uint256 => string) public productImages;

    // Mapping of products to their respective categories
    mapping (uint256 => string) public productCategories;

    // Mapping of products to their respective ratings
    mapping (uint256 => uint256) public productRatings;

    // Mapping of sellers to their respective ratings
    mapping (address => uint256) public sellerRatings;

    // Mapping of buyers to their respective ratings
    mapping (address => uint256) public buyerRatings;

    // Event emitted when a product is listed
    event ProductListed(uint256 indexed productId, address indexed seller, uint256 price, string description, string image, string category);

    // Event emitted when a product is purchased
    event ProductPurchased(uint256 indexed productId, address indexed buyer, address indexed seller, uint256 price);

    // Event emitted when a product is rated
    event ProductRated(uint256 indexed productId, address indexed rater, uint256 rating);

    // Event emitted when a seller is rated
    event SellerRated(address indexed seller, address indexed rater, uint256 rating);

    // Event emitted when a buyer is rated
    event BuyerRated(address indexed buyer, address indexed rater, uint256 rating);

    // Event emitted when a dispute is raised
    event DisputeRaised(uint256 indexed productId, address indexed buyer, address indexed seller, string reason);

    // Event emitted when a dispute is resolved
    event DisputeResolved(uint256 indexed productId, address indexed buyer, address indexed seller, string resolution);

    // Reputation system
    ReputationSystem public reputationSystem;

    // Dispute resolution mechanism
    DisputeResolution public disputeResolution;

    // Constructor
    constructor() public {
        reputationSystem = new ReputationSystem();
        disputeResolution = new DisputeResolution();
    }

    // Function to list a product
    function listProduct(uint256 _productId, uint256 _price, string memory _description, string memory _image, string memory _category) public {
        // Check if the product is already listed
        require(productSellers[_productId] == address(0), "Product is already listed");

        // Set the product seller
        productSellers[_productId] = msg.sender;

        // Set the product price
        productPrices[_productId] = _price;

        // Set the product description
        productDescriptions[_productId] = _description;

        // Set the product image
        productImages[_productId] = _image;

        // Set the product category
        productCategories[_productId] = _category;

        // Emit the ProductListed event
        emit ProductListed(_productId, msg.sender, _price, _description, _image, _category);
    }

    // Function to purchase a product
    function purchaseProduct(uint256 _productId) public payable {
        // Check if the product is listed
        require(productSellers[_productId] != address(0), "Product is not listed");

        // Check if the buyer has enough funds
        require(msg.value >= productPrices[_productId], "Insufficient funds");

        // Set the buyer's purchase
        buyerPurchases[msg.sender].push(_productId);

        // Set the seller's sale
        sellerProducts[productSellers[_productId]].push(_productId);

        // Transfer the funds to the seller
        payable(productSellers[_productId]).transfer(msg.value);

        // Emit the ProductPurchased event
        emit ProductPurchased(_productId, msg.sender, productSellers[_productId], productPrices[_productId]);
    }

        // Function to rate a product
    function rateProduct(uint256 _productId, uint256 _rating) public {
        // Check if the product is listed
        require(productSellers[_productId] != address(0), "Product is not listed");

        // Check if the rater has purchased the product
        require(buyerPurchases[msg.sender].contains(_productId), "Rater has not purchased the product");

        // Set the product rating
        productRatings[_productId] = _rating;

        // Emit the ProductRated event
        emit ProductRated(_productId, msg.sender, _rating);
    }

    // Function to rate a seller
    function rateSeller(address _seller, uint256 _rating) public {
        // Check if the seller is registered
        require(sellerProducts[_seller].length > 0, "Seller is not registered");

        // Check if the rater has purchased from the seller
        require(buyerPurchases[msg.sender].contains(sellerProducts[_seller][0]), "Rater has not purchased from the seller");

        // Set the seller rating
        sellerRatings[_seller] = _rating;

        // Emit the SellerRated event
        emit SellerRated(_seller, msg.sender, _rating);
    }

    // Function to rate a buyer
    function rateBuyer(address _buyer, uint256 _rating) public {
        // Check if the buyer is registered
        require(buyerPurchases[_buyer].length > 0, "Buyer is not registered");

        // Check if the rater has sold to the buyer
        require(sellerProducts[msg.sender].contains(buyerPurchases[_buyer][0]), "Rater has not sold to the buyer");

        // Set the buyer rating
        buyerRatings[_buyer] = _rating;

        // Emit the BuyerRated event
        emit BuyerRated(_buyer, msg.sender, _rating);
    }

    // Function to raise a dispute
    function raiseDispute(uint256 _productId, string memory _reason) public {
        // Check if the product is listed
        require(productSellers[_productId] != address(0), "Product is not listed");

        // Check if the disputant has purchased the product
        require(buyerPurchases[msg.sender].contains(_productId), "Disputant has not purchased the product");

        // Create a new dispute
        disputeResolution.createDispute(_productId, msg.sender, productSellers[_productId], _reason);

        // Emit the DisputeRaised event
        emit DisputeRaised(_productId, msg.sender, productSellers[_productId], _reason);
    }

    // Function to resolve a dispute
    function resolveDispute(uint256 _productId, string memory _resolution) public {
        // Check if the dispute exists
        require(disputeResolution.disputes[_productId].exists, "Dispute does not exist");

        // Resolve the dispute
        disputeResolution.resolveDispute(_productId, _resolution);

        // Emit the DisputeResolved event
        emit DisputeResolved(_productId, disputeResolution.disputes[_productId].buyer, disputeResolution.disputes[_productId].seller, _resolution);
    }
}
