import { test, expect } from '@jest/globals';
import { Product } from '../models/productModel';
import { productReducer } from '../reducers/productReducer';

describe('Product Tests', () => {
  test('Create a new product', async () => {
    const product = new Product({
      name: 'Product 1',
      description: 'This is a product',
      price: 10.99,
    });
    expect(product).toHaveProperty('id');
    expect(product.name).toBe('Product 1');
    expect(product.description).toBe('This is a product');
    expect(product.price).toBe(10.99);
  });

  test('Product reducer should handle SET_PRODUCTS action', async () => {
    const initialState = [];
    const action = { type: 'SET_PRODUCTS', payload: [{ name: 'Product 1', description: 'This is a product', price: 10.99 }] };
    const state = productReducer(initialState, action);
    expect(state).toHaveLength(1);
    expect(state[0]).toHaveProperty('name', 'Product 1');
    expect(state[0]).toHaveProperty('description', 'This is a product');
    expect(state[0]).toHaveProperty('price', 10.99);
  });

  test('Product reducer should handle SET_PRODUCT action', async () => {
    const initialState = {};
    const action = { type: 'SET_PRODUCT', payload: { name: 'Product 1', description: 'This is a product', price: 10.99 } };
    const state = productReducer(initialState, action);
    expect(state).toHaveProperty('name', 'Product 1');
    expect(state).toHaveProperty('description', 'This is a product');
    expect(state).toHaveProperty('price', 10.99);
  });
});
