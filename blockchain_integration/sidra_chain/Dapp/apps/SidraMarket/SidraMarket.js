const fulfillOrder = (id, buyer) => {
    const web3 = new Web3(window.ethereum);
    const contract = new web3.eth.Contract(SidraMarket.abi, SidraMarket.address);

    contract.methods.fulfillOrder(id, buyer).send({ from: window.ethereum.selectedAddress });
};

const raiseDispute = (id, buyer) => {
    const web3 = new Web3(window.ethereum);
    const contract = new web3.eth.Contract(SidraMarket.abi, SidraMarket.address);

    contract.methods.raiseDispute(id, buyer).send({ from: window.ethereum.selectedAddress });
};

const resolveDispute = (id, buyer) => {
    const web3 = new Web3(window.ethereum);
    const contract = new web3.eth.Contract(SidraMarket.abi, SidraMarket.address);

    contract.methods.resolveDispute(id, buyer).send({ from: window.ethereum.selectedAddress });
};

return (
    <div>
        <h1>Sidra Market</h1>
        <ul>
            {products.map((product) => (
                <li key={product.id}>
                    <h2>{product.name}</h2>
                    <p>{product.description}</p>
                    <p>Price: {product.price}</p>
                    <button onClick={() => addProduct(product.id, product.name, product.description, product.price)}>Add to Cart</button>
                </li>
            ))}
        </ul>
        <ul>
            {orders.map((order) => (
                <li key={order.id}>
                    <h2>Order {order.id}</h2>
                    <p>Buyer: {order.buyer}</p>
                    <p>Product: {order.productId}</p>
                    <p>Quantity: {order.quantity}</p>
                    <p>Total Price: {order.totalPrice}</p>
                    <button onClick={() => fulfillOrder(order.id, order.buyer)}>Fulfill Order</button>
                    <button onClick={() => raiseDispute(order.id, order.buyer)}>Raise Dispute</button>
                </li>
            ))}
        </ul>
        <ul>
            {Object.keys(reputationScores).map((address) => (
                <li key={address}>
                    <h2>Reputation Score for {address}</h2>
                    <p>Score: {reputationScores[address]}</p>
                </li>
            ))}
        </ul>
    </div>
);
