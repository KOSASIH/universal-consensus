import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const secretKey = process.env.SECRET_KEY;

export function generateToken(user) {
  const payload = {
    id: user.id,
    email: user.email,
    role: user.role,
  };
  return jwt.sign(payload, secretKey, { expiresIn: '1h' });
}

export function verifyToken(token) {
  try {
    const decoded = jwt.verify(token, secretKey);
    return decoded;
  } catch (error) {
    return null;
  }
}

export function hashPassword(password) {
  return bcrypt.hashSync(password, 10);
}

export function comparePassword(password, hashedPassword) {
  return bcrypt.compareSync(password, hashedPassword);
}

export function authenticateUser(email, password, users) {
  const user = users.find((user) => user.email === email);
  if (!user) return null;
  const isValid = comparePassword(password, user.password);
  if (!isValid) return null;
  return user;
}
