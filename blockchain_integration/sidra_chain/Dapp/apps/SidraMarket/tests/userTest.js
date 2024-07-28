import { test, expect } from '@jest/globals';
import { User } from '../models/userModel';
import { generateToken, verifyToken } from '../utils/auth';
import { userReducer } from '../reducers/userReducer';

describe('User Tests', () => {
  test('Create a new user', async () => {
    const user = new User({
      name: 'John Doe',
      email: 'john.doe@example.com',
      password: 'password123',
    });
    expect(user).toHaveProperty('id');
    expect(user.name).toBe('John Doe');
    expect(user.email).toBe('john.doe@example.com');
  });

  test('Generate a token for a user', async () => {
    const user = new User({
      name: 'John Doe',
      email: 'john.doe@example.com',
      password: 'password123',
    });
    const token = generateToken(user);
    expect(token).toBeInstanceOf(String);
  });

  test('Verify a token for a user', async () => {
    const user = new User({
      name: 'John Doe',
      email: 'john.doe@example.com',
      password: 'password123',
    });
    const token = generateToken(user);
    const verifiedUser = verifyToken(token);
    expect(verifiedUser).toHaveProperty('id');
    expect(verifiedUser.name).toBe('John Doe');
    expect(verifiedUser.email).toBe('john.doe@example.com');
  });

  test('User reducer should handle SET_USER action', async () => {
    const initialState = {};
    const action = { type: 'SET_USER', payload: { name: 'John Doe', email: 'john.doe@example.com' } };
    const state = userReducer(initialState, action);
    expect(state).toHaveProperty('name', 'John Doe');
    expect(state).toHaveProperty('email', 'john.doe@example.com');
  });
});
