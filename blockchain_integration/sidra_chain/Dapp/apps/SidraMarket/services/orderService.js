const Order = require('../models/Order');
const Product = require('../models/Product');

const createOrder = async (req, res) => {
  const order = new Order(req.body);
  await order.save();
  res.send({ message: 'Order created successfully' });
};

const getAllOrders = async (req, res) => {
  const orders = await Order.find();
  res.send(orders);
};

const getOrderById = async (req, res) => {
  const order = await Order.findById(req.params.id);
  res.send(order);
};

const updateOrder = async (req, res) => {
  const order = await Order.findById(req.params.id);
  order.products = req.body.products;
  await order.save();
  res.send({ message: 'Order updated successfully' });
};

const deleteOrder = async (req, res) => {
  await Order.findByIdAndRemove(req.params.id);
  res.send({ message: 'Order deleted successfully' });
};

module.exports = { createOrder, getAllOrders, getOrderById, updateOrder, deleteOrder };
