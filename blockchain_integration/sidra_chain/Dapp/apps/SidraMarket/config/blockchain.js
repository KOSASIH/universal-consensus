const Web3 = require('web3');
const contract = require('truffle-contract');

const blockchainConfig = {
  provider: 'http://localhost:8545',
  contractAddress: '0x1234567890abcdef',
};

const web3 = new Web3(new Web3.providers.HttpProvider(blockchainConfig.provider));
const contractInstance = contract(blockchainConfig.contractAddress);

module.exports = { web3, contractInstance };
