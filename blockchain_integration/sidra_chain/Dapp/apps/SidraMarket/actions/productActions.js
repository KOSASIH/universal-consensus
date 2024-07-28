import axios from 'axios';
import { SET_PRODUCTS, SET_PRODUCT, SET_PRODUCT_ERROR } from '../constants/productConstants';

export const fetchProducts = () => async (dispatch) => {
  try {
    const response = await axios.get('/api/products');
    dispatch({ type: SET_PRODUCTS, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const createProduct = (name, description, price) => async (dispatch) => {
  try {
    const response = await axios.post('/api/products', { name, description, price });
    dispatch({ type: SET_PRODUCT, payload: response.data });
  } catch (error) {
    dispatch({ type: SET_PRODUCT_ERROR, payload: error.response.data });
  }
};

export const updateProduct = (id, name, description, price) => async (dispatch) => {
  try {
    const response = await axios.put(`/api/products/${id}`, { name, description, price });
    dispatch({ type: SET_PRODUCT, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const deleteProduct = (id) => async (dispatch) => {
  try {
    await axios.delete(`/api/products/${id}`);
    dispatch({ type: SET_PRODUCTS, payload: [] });
  } catch (error) {
    console.error(error);
  }
};

export const fetchProduct = (id) => async (dispatch) => {
  try {
    const response = await axios.get(`/api/products/${id}`);
    dispatch({ type: SET_PRODUCT, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};
