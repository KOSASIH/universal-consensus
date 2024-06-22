// blockchain-network-interface/main.go
package main

import (
	"fmt"
	"log"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/blockchain"
)

func main() {
	config, err := blockchain.LoadConfig("config.json")
	if err != nil {
		log.Fatal(err)
	}

	hyperledgerBlockchain := &blockchain.HyperledgerBlockchain{
		client: blockchain.NewHyperledgerClient(config.Hyperledger.URL, config.Hyperledger.Username, config.Hyperledger.Password),
	}

	ethereumBlockchain := &blockchain.EthereumBlockchain{
		client: blockchain.NewEthereumClient(config.Ethereum.URL, config.Ethereum.Username, config.Ethereum.Password),
	}

	quorumBlockchain := &blockchain.QuorumBlockchain{
		client: blockchain.NewQuorumClient(config.Quorum.URL, config.Quorum.Username, config.Quorum.Password),
	}

	fmt.Println("Blockchain Network Interface")
	fmt.Println("----------------------------")

	fmt.Println("Deploying contract to Hyperledger...")
	contractID, err := hyperledgerBlockchain.DeployContract("path/to/contract.sol")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Contract ID:", contractID)

	fmt.Println("Invoking contract on Ethereum...")
	result, err := ethereumBlockchain.InvokeContract(contractID, "function", []byte("arg1", "arg2"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Result:", result)

	fmt.Println("Querying contract on Quorum...")
	result, err = quorumBlockchain.QueryContract(contractID, "function", []byte("arg1", "arg2"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Result:", result)
}
