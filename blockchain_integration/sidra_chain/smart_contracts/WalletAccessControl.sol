pragma solidity ^0.8.0;

contract WalletAccessControl is Owner {
    mapping(address => bool) public enabledWallets;

    event WalletEnabled(address indexed wallet);
    event WalletDisabled(address indexed wallet);

    function enableWallet(address wallet) public onlyOwner {
        enabledWallets[wallet] = true;
        emit WalletEnabled(wallet);
    }

    function disableWallet(address wallet) public onlyOwner {
        enabledWallets[wallet] = false;
        emit WalletDisabled(wallet);
    }

    function isWalletEnabled(address wallet) public view returns (bool) {
        return enabledWallets[wallet];
    }
}
