package main

import (
	"fmt"
	"log"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/blockchain"
)

func main() {
	// Create a new blockchain
	bc := blockchain.NewBlockchain()

	// Create a new block
	tx1 := block.Transaction{
		ID:        "tx1",
		Timestamp: time.Now(),
		Sender:    "Alice",
		Recipient: "Bob",
		Amount:    10,
	}
	tx2 := block.Transaction{
		ID:        "tx2",
		Timestamp: time.Now(),
		Sender:    "Bob",
		Recipient: "Charlie",
		Amount:    20,
	}
	block := block.NewBlock([]block.Transaction{tx1, tx2}, "")

	// Add the block to the blockchain
	err := bc.AddBlock(block)
	if err!= nil {
		log.Fatal(err)
	}

	fmt.Println("Blockchain state:")
	fmt.Println(bc.GetChain())

	// Mine a new block
	tx3 := block.Transaction{
		ID:        "tx3",
		Timestamp: time.Now(),
		Sender:    "Charlie",
		Recipient: "Alice",
		Amount:    30,
	}
	newBlock := block.NewBlock([]block.Transaction{tx3}, bc.GetCurrentBlock().Header.Hash)

	// Add the new block to the blockchain
	err = bc.AddBlock(newBlock)
	if err!= nil {
		log.Fatal(err)
	}

	fmt.Println("Updated blockchain state:")
	fmt.Println(bc.GetChain())
}
