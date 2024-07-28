import { SET_USER, SET_AUTH_ERROR, CLEAR_AUTH_ERROR } from '../constants/userConstants';

const initialState = {
  user: null,
  authError: null,
};

export default function userReducer(state = initialState, action) {
  switch (action.type) {
    case SET_USER:
      return { ...state, user: action.payload };
    case SET_AUTH_ERROR:
      return { ...state, authError: action.payload };
    case CLEAR_AUTH_ERROR:
      return { ...state, authError: null };
    default:
      return state;
  }
}
