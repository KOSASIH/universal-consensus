import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import contract from './contracts/SidraMarket.json';

const contractAddress = "0x..."; // replace with the actual contract address
const abi = contract.abi;

function App() {
  const [account, setAccount] = useState('');
  const [sidraMarket, setSidraMarket] = useState(null);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [rating, setRating] = useState(0);
  const [review, setReview] = useState('');
  const [dispute, setDispute] = useState(null);

  useEffect(() => {
    const loadWeb3 = async () => {
      if (window.ethereum) {
        const web3 = new Web3(window.ethereum);
        await window.ethereum.enable();
        setAccount(await web3.eth.getAccounts()[0]);
        setSidraMarket(new web3.eth.Contract(abi, contractAddress));
      }
    };
    loadWeb3();
  }, []);

  useEffect(() => {
    const loadProducts = async () => {
      if (sidraMarket) {
        const products = await sidraMarket.methods.getProducts().call();
        setProducts(products);
      }
    };
    loadProducts();
  }, [sidraMarket]);

  const handlePurchase = async (productId) => {
    await sidraMarket.methods.purchaseProduct(productId).send({ from: account });
  };

  const handleRateProduct = async (productId, rating) => {
    await sidraMarket.methods.rateProduct(productId, rating).send({ from: account });
  };

  const handleReviewProduct = async (productId, review) => {
    await sidraMarket.methods.reviewProduct(productId, review).send({ from: account });
  };

  const handleRaiseDispute = async (productId, reason) => {
    await sidraMarket.methods.raiseDispute(productId, reason).send({ from: account });
  };

  const handleResolveDispute = async (disputeId, resolution) => {
    await sidraMarket.methods.resolveDispute(disputeId, resolution).send({ from: account });
  };

  return (
    <div className="main-app">
      <h1>Sidra Market</h1>
      <div>
        {products.map((product) => (
          <div key={product.id}>
            <h2>{product.name}</h2>
            <p>{product.description}</p>
            <p>Price: {product.price} ETH</p>
            <button onClick={() => handlePurchase(product.id)}>Purchase</button>
            <button onClick={() => handleRateProduct(product.id, 5)}>Rate 5 stars</button>
            <button onClick={() => handleReviewProduct(product.id, 'Great product!')}>Review</button>
            <button onClick={() => handleRaiseDispute(product.id, 'Reason for dispute')}>Raise Dispute</button>
          </div>
        ))}
      </div>
      {selectedProduct && (
        <div>
          <h2>Selected Product</h2>
          <p>{selectedProduct.name}</p>
          <p>{selectedProduct.description}</p>
          <p>Price: {selectedProduct.price} ETH</p>
          <button onClick={() => handlePurchase(selectedProduct.id)}>Purchase</button>
          <button onClick={() => handleRateProduct(selectedProduct.id, 5)}>Rate 5 stars</button>
          <button onClick={() => handleReviewProduct(selectedProduct.id, 'Great product!')}>Review</button>
          <button onClick={() => handleRaiseDispute(selectedProduct.id, 'Reason for dispute')}>Raise Dispute</button>
        </div>
      )}
      {dispute && (
        <div>
          <h2>Dispute</h2>
          <p>{dispute.reason}</p>
          <button onClick={() => handleResolveDispute(dispute.id, 'Resolved')}>Resolve</button>
        </div>
      )}
    </div>
  );
}

export default App;
