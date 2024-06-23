package main

import (
	"fmt"
	"log"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain"
)

func main() {
	// Initialize the OmniaChain system
	oc, err := OmniaChain.NewOmniaChain()
	if err != nil {
		log.Fatal(err)
	}

	// Start the OmniaChain system
	err = oc.Start()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("OmniaChain system started")

	// Create a new transaction
	tx, err := OmniaChain.NewTransaction("Alice", "Bob", big.NewInt(10), nil)
	if err != nil {
		log.Fatal(err)
	}

	// Add the transaction to the transaction pool
	err = oc.GetTxPool().AddTx(tx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Transaction added to pool")

	// Mine a new block
	block, err := oc.GetChain().MineBlock(oc.GetTxPool().GetTxs())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("New block mined:", block.Header.Hash)

	// Add the block to the blockchain
	err = oc.GetChain().AddBlock(block)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Block added to blockchain")

	// Print the current blockchain state
	fmt.Println("Blockchain state:")
	fmt.Println(oc.GetChain().String())
}
