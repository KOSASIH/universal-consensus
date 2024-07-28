const User = require('../models/User');

const registerUser = async (req, res) => {
  const user = new User(req.body);
  await user.save();
  res.send({ message: 'User registered successfully' });
};

const loginUser = async (req, res) => {
  const user = await User.findOne({ email: req.body.email });
  if (!user) {
    return res.status(401).send({ message: 'Invalid email or password' });
  }
  const isValidPassword = await user.comparePassword(req.body.password);
  if (!isValidPassword) {
    return res.status(401).send({ message: 'Invalid email or password' });
  }
  res.send({ message: 'User logged in successfully' });
};

const getUserProfile = async (req, res) => {
  const user = await User.findById(req.user.id);
  res.send(user);
};

module.exports = { registerUser, loginUser, getUserProfile };
