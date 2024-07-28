const express = require('express');
const router = express.Router();
const orderService = require('../services/orderService');

router.post('/create', orderService.createOrder);
router.get('/all', orderService.getAllOrders);
router.get('/:id', orderService.getOrderById);

module.exports = router;
