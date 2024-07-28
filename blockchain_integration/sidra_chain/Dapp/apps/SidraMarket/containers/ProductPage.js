import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchProducts } from '../actions/productActions';
import ProductList from '../components/ProductList';

const ProductPage = () => {
  const [products, setProducts] = useState([]);
  const dispatch = useDispatch();

  useEffect(() => {
    const fetchProductData = async () => {
      try {
        const productData = await dispatch(fetchProducts());
        setProducts(productData);
      } catch (error) {
        console.error(error);
      }
    };
    fetchProductData();
  }, [dispatch]);

  return (
    <div>
      <h1>Product Page</h1>
      <ProductList products={products} />
    </div>
  );
};

export default ProductPage;
