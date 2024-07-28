import { SET_ORDERS, SET_ORDER, SET_ORDER_ERROR } from '../constants/orderConstants';

const initialState = {
  orders: [],
  order: null,
  orderError: null,
};

export default function orderReducer(state = initialState, action) {
  switch (action.type) {
    case SET_ORDERS:
      return { ...state, orders: action.payload };
    case SET_ORDER:
      return { ...state, order: action.payload };
    case SET_ORDER_ERROR:
      return { ...state, orderError: action.payload };
    default:
      return state;
  }
}
