const express = require('express');
const router = express.Router();
const productService = require('../services/productService');

router.post('/create', productService.createProduct);
router.get('/all', productService.getAllProducts);
router.get('/:id', productService.getProductById);

module.exports = router;
