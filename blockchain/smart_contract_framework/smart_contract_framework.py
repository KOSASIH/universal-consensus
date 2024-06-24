from evm import EVM

class SmartContractFramework:
    def __init__(self, evm_config):
        self.evm = EVM(evm_config)

    def deploy_contract(self, contract_code):
        # Compile and deploy the contract
        contract_bytecode = self.evm.compile(contract_code)
        contract_address = self.evm.deploy(contract_bytecode)
        return contract_address

    def execute_contract(self, contract_address, function, args):
        # Execute a contract function
        result = self.evm.execute(contract_address, function, args)
        return result
