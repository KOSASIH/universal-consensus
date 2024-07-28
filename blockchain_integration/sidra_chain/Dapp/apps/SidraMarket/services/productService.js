const Product = require('../models/Product');

const createProduct = async (req, res) => {
  try {
    const product = new Product(req.body);
    await product.save();
    res.send({ message: 'Product created successfully' });
  } catch (error) {
    res.status(500).send({ message: 'Error creating product' });
  }
};

const getAllProducts = async (req, res) => {
  try {
    const products = await Product.find();
    res.send(products);
  } catch (error) {
    res.status(500).send({ message: 'Error fetching products' });
  }
};

const getProductById = async (req, res) => {
  try {
    const product = await Product.findById(req.params.id);
    if (!product) {
      return res.status(404).send({ message: 'Product not found' });
    }
    res.send(product);
  } catch (error) {
    res.status(500).send({ message: 'Error fetching product' });
  }
};

const updateProduct = async (req, res) => {
  try {
    const product = await Product.findById(req.params.id);
    if (!product) {
      return res.status(404).send({ message: 'Product not found' });
    }
    product.name = req.body.name;
    product.description = req.body.description;
    product.price = req.body.price;
    await product.save();
    res.send({ message: 'Product updated successfully' });
  } catch (error) {
    res.status(500).send({ message: 'Error updating product' });
  }
};

const deleteProduct = async (req, res) => {
  try {
    await Product.findByIdAndRemove(req.params.id);
    res.send({ message: 'Product deleted successfully' });
  } catch (error) {
    res.status(500).send({ message: 'Error deleting product' });
  }
};

module.exports = { createProduct, getAllProducts, getProductById, updateProduct, deleteProduct };
