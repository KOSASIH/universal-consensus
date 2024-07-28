import axios from 'axios';
import { SET_USER, SET_AUTH_ERROR, CLEAR_AUTH_ERROR } from '../constants/userConstants';

export const login = (email, password) => async (dispatch) => {
  try {
    const response = await axios.post('/api/login', { email, password });
    dispatch({ type: SET_USER, payload: response.data });
  } catch (error) {
    dispatch({ type: SET_AUTH_ERROR, payload: error.response.data });
  }
};

export const logout = () => async (dispatch) => {
  try {
    await axios.post('/api/logout');
    dispatch({ type: SET_USER, payload: null });
  } catch (error) {
    console.error(error);
  }
};

export const register = (name, email, password) => async (dispatch) => {
  try {
    const response = await axios.post('/api/register', { name, email, password });
    dispatch({ type: SET_USER, payload: response.data });
  } catch (error) {
    dispatch({ type: SET_AUTH_ERROR, payload: error.response.data });
  }
};

export const updateUser = (id, name, email) => async (dispatch) => {
  try {
    const response = await axios.put(`/api/users/${id}`, { name, email });
    dispatch({ type: SET_USER, payload: response.data });
  } catch (error) {
    console.error(error);
  }
};

export const clearAuthError = () => (dispatch) => {
  dispatch({ type: CLEAR_AUTH_ERROR });
};
