const express = require('express');
const app = express();
const Web3 = require('web3');
const sidraMarketToken = require('./SidraMarketToken.sol');

app.use(express.json());

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const tokenContract = new web3.eth.Contract(sidraMarketToken.abi, '0x...TokenContractAddress...');

app.post('/mint', async (req, res) => {
  try {
    const { account } = req.body;
    await tokenContract.methods.mint(account, 10).send({ from: account });
    res.json({ message: 'Tokens minted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error minting tokens' });
  }
});

app.post('/transfer', async (req, res) => {
  try {
    const { from, to, amount } = req.body;
    await tokenContract.methods.transfer(from, to, amount).send({ from: from });
    res.json({ message: 'Tokens transferred successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error transferring tokens' });
  }
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
