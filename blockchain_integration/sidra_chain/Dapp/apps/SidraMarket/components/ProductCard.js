import React from 'react';
import { Link } from 'react-router-dom';
import { FaShoppingCart } from 'react-icons/fa';

const ProductCard = ({ product }) => {
  return (
    <div className="product-card">
      <img src={product.image} alt={product.name} />
      <h2>
        <Link to={`/products/${product.id}`}>{product.name}</Link>
      </h2>
      <p>{product.description}</p>
      <p>
        Price: <span className="product-card__price">{product.price}</span>
      </p>
      <button className="product-card__btn">
        <FaShoppingCart /> Add to Cart
      </button>
    </div>
  );
};

export default ProductCard;
