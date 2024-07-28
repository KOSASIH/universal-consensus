import crypto from 'crypto';

export function generateRandomBytes(size) {
  return crypto.randomBytes(size);
}

export function hashData(data, algorithm = 'sha256') {
  return crypto.createHash(algorithm).update(data).digest('hex');
}

export function encryptData(data, key) {
  const cipher = crypto.createCipher('aes-256-cbc', key);
  let encrypted = cipher.update(data, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  return encrypted;
}

export function decryptData(encrypted, key) {
  const decipher = crypto.createDecipher('aes-256-cbc', key);
  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  return decrypted;
}

export function signData(data, privateKey) {
  const signer = crypto.createSign('RSA-SHA256');
  signer.update(data);
  signer.end();
  return signer.sign(privateKey, 'hex');
}

export function verifySignature(data, signature, publicKey) {
  const verifier = crypto.createVerify('RSA-SHA256');
  verifier.update(data);
  verifier.end();
  return verifier.verify(publicKey, signature, 'hex');
}
