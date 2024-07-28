import { test, expect } from '@jest/globals';
import { Order } from '../models/orderModel';
import { orderReducer } from '../reducers/orderReducer';

describe('Order Tests', () => {
  test('Create a new order', async () => {
    const order = new Order({
      userId: '123',
      products: [{ id: '1', quantity: 2 }],
      total: 21.98,
    });
    expect(order).toHaveProperty('id');
    expect(order.userId).toBe('123');
    expect(order.products).toHaveLength(1);
    expect(order.total).toBe(21.98);
  });

    test('Order reducer should handle SET_ORDERS action', async () => {
    const initialState = [];
    const action = { type: 'SET_ORDERS', payload: [{ userId: '123', products: [{ id: '1', quantity: 2 }], total: 21.98 }] };
    const state = orderReducer(initialState, action);
    expect(state).toHaveLength(1);
    expect(state[0]).toHaveProperty('userId', '123');
    expect(state[0]).toHaveProperty('products', [{ id: '1', quantity: 2 }]);
    expect(state[0]).toHaveProperty('total', 21.98);
  });

  test('Order reducer should handle SET_ORDER action', async () => {
    const initialState = {};
    const action = { type: 'SET_ORDER', payload: { userId: '123', products: [{ id: '1', quantity: 2 }], total: 21.98 } };
    const state = orderReducer(initialState, action);
    expect(state).toHaveProperty('userId', '123');
    expect(state).toHaveProperty('products', [{ id: '1', quantity: 2 }]);
    expect(state).toHaveProperty('total', 21.98);
  });

  test('Order reducer should handle UPDATE_ORDER action', async () => {
    const initialState = { userId: '123', products: [{ id: '1', quantity: 2 }], total: 21.98 };
    const action = { type: 'UPDATE_ORDER', payload: { products: [{ id: '2', quantity: 3 }] } };
    const state = orderReducer(initialState, action);
    expect(state).toHaveProperty('userId', '123');
    expect(state).toHaveProperty('products', [{ id: '2', quantity: 3 }]);
    expect(state).toHaveProperty('total', 32.97);
  });

  test('Order reducer should handle DELETE_ORDER action', async () => {
    const initialState = [{ userId: '123', products: [{ id: '1', quantity: 2 }], total: 21.98 }];
    const action = { type: 'DELETE_ORDER', payload: '123' };
    const state = orderReducer(initialState, action);
    expect(state).toHaveLength(0);
  });
});
