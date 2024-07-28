import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchOrders } from '../actions/orderActions';
import OrderList from '../components/OrderList';

const OrderPage = () => {
  const [orders, setOrders] = useState([]);
  const dispatch = useDispatch();

  useEffect(() => {
    const fetchOrderData = async () => {
      try {
        const orderData = await dispatch(fetchOrders());
        setOrders(orderData);
      } catch (error) {
        console.error(error);
      }
    };
    fetchOrderData();
  }, [dispatch]);

  return (
    <div>
      <h1>Order Page</h1>
      <OrderList orders={orders} />
    </div>
  );
};

export default OrderPage;
