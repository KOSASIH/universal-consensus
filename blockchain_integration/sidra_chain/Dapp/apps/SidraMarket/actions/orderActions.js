import axios from 'axios';
import { SET_ORDERS, SET_ORDER, SET_ORDER_ERROR } from '../constants/orderConstants';

export const fetchOrders = () => async (dispatch) => {
  try {
    const response = await axios.get('/api/orders');
    dispatch({ type: SET_ORDERS, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const createOrder = (products, total) => async (dispatch) => {
  try {
    const response = await axios.post('/api/orders', { products, total });
    dispatch({ type: SET_ORDER, payload: response.data });
  } catch (error) {
    dispatch({ type: SET_ORDER_ERROR, payload: error.response.data });
  }
};

export const updateOrder = (id, products, total) => async (dispatch) => {
  try {
    const response = await axios.put(`/api/orders/${id}`, { products, total });
    dispatch({ type: SET_ORDER, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const deleteOrder = (id) => async (dispatch) => {
  try {
    await axios.delete(`/api/orders/${id}`);
    dispatch({ type: SET_ORDERS, payload: [] });
  } catch (error) {
    console.error(error);
  }
};

export const updateOrderStatus = (id, status) => async (dispatch) => {
  try {
    const response = await axios.put(`/api/orders/${id}/status`, { status });
    dispatch({ type: SET_ORDER, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const cancelOrder = (id) => async (dispatch) => {
  try {
    await axios.delete(`/api/orders/${id}`);
    dispatch({ type: SET_ORDERS, payload: [] });
  } catch (error) {
    console.error(error);
  }
};
