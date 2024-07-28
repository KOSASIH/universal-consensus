import { SET_PRODUCTS, SET_PRODUCT, SET_PRODUCT_ERROR } from '../constants/productConstants';

const initialState = {
  products: [],
  product: null,
  productError: null,
};

export default function productReducer(state = initialState, action) {
  switch (action.type) {
    case SET_PRODUCTS:
      return { ...state, products: action.payload };
    case SET_PRODUCT:
      return { ...state, product: action.payload };
    case SET_PRODUCT_ERROR:
      return { ...state, productError: action.payload };
    default:
      return state;
  }
}
