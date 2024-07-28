[
    {
        "constant": true,
        "inputs": [],
        "name": "getProducts",
        "outputs": [
            {
                "name": "",
                "type": "uint256[]"
            }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [],
        "name": "getOrders",
        "outputs": [
            {
                "name": "",
                "type": "uint256[]"
            }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [],
        "name": "getReputationScores",
        "outputs": [
            {
                "name": "",
                "type": "uint256[]"
            }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        [
    {
        "constant": false,
        "inputs": [
            {
                "name": "_id",
                "type": "uint256"
            },
            {
                "name": "_name",
                "type": "string"
            },
            {
                "name": "_description",
                "type": "string"
            },
            {
                "name": "_price",
                "type": "uint256"
            }
        ],
        "name": "addProduct",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "name": "_id",
                "type": "uint256"
            },
            {
                "name": "_buyer",
                "type": "address"
            },
            {
                "name": "_productId",
                "type": "uint256"
            },
            {
                "name": "_quantity",
                "type": "uint256"
            }
        ],
        "name": "placeOrder",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "name": "_id",
                "type": "uint256"
            },
            {
                "name": "_buyer",
                "type": "address"
            }
        ],
        "name": "fulfillOrder",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "name": "_id",
                "type": "uint256"
            },
            {
                "name": "_buyer",
                "type": "address"
            }
        ],
        "name": "raiseDispute",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "name": "_id",
                "type": "uint256"
            },
            {
                "name": "_buyer",
                "type": "address"
            }
        ],
        "name": "resolveDispute",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "name": "_owner",
                "type": "address"
            },
            {
                "indexed": true,
                "name": "_id",
                "type": "uint256"
            }
        ],
        "name": "ProductAdded",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "name": "_buyer",
                "type": "address"
            },
            {
                "indexed": true,
                "name": "_id",
                "type": "uint256"
            }
        ],
        "name": "OrderPlaced",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "name": "_buyer",
                "type": "address"
            },
            {
                "indexed": true,
                "name": "_id",
                "type": "uint256"
            }
        ],
        "name": "OrderFulfilled",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "name": "_buyer",
                "type": "address"
            },
            {
                "indexed": true,
                "name": "_id",
                "type": "uint256"
            }
        ],
        "name": "DisputeRaised",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "name": "_buyer",
                "type": "address"
            },
            {
                "indexed": true,
                "name": "_id",
                "type": "uint256"
            }
        ],
        "name": "DisputeResolved",
        "type": "event"
    }
]
