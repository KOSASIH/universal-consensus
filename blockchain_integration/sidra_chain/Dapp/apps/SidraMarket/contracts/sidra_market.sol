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
