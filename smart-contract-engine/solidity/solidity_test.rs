// smart-contract-engine/solidity/solidity_test.rs
use solidity::{Solidity, Contract, Abi, AbiInput, AbiOutput};

#[test]
fn test_solidity_compile() {
    let mut solidity = Solidity::new();
    let source_code = "pragma solidity ^0.8.0; contract TestContract { function test() public { } }";
    assert!(solidity.compile(source_code).is_ok());
}

#[test]
fn test_solidity_deploy() {
    let mut solidity = Solidity::new();
    let contract_name = "TestContract";
    assert!(solidity.deploy(contract_name).is_ok());
}

#[test]
fn test_solidity_execute() {
    let mut solidity = Solidity::new();
    let contract_name = "TestContract";
    let function_name = "test";
    let args = vec![];
    assert!(solidity.execute(contract_name, function_name, args).is_ok());
}
