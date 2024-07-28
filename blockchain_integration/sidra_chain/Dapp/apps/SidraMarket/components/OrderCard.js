import React from 'react';
import { Link } from 'react-router-dom';

const OrderCard = ({ order }) => {
  return (
    <div className="order-card">
      <h2>
        <Link to={`/orders/${order.id}`}>Order #{order.id}</Link>
      </h2>
      <p>Order Date: {order.date}</p>
      <p>Order Total: {order.total}</p>
      <ul>
        {order.items.map((item, index) => (
          <li key={index}>
            <span>{item.name}</span>
            <span>x {item.quantity}</span>
            <span>${item.price}</span>
          </li>
        ))}
      </ul>
      <button className="order-card__btn">View Order Details</button>
    </div>
  );
};

export default OrderCard;
