package main

import (
	"context"
	"crypto/ecdsa"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net"
	"sync"
	"time"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/chain"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/consensus"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/network"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/node"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/transaction"
)

type OmniaChain struct {
	chain    *chain.Chain
	network  *network.Network
	consensus *consensus.Consensus
	nodes    []*node.Node
	txPool   *transaction.TxPool
}

func NewOmniaChain() *OmniaChain {
	return &OmniaChain{
		chain:    chain.NewChain(),
		network:  network.NewNetwork(),
		consensus: consensus.NewConsensus(),
		nodes:    make([]*node.Node, 0),
		txPool:   transaction.NewTxPool(),
	}
}

func (oc *OmniaChain) Start() error {
	// Start the network
	go oc.network.Start()

	// Start the consensus algorithm
	go oc.consensus.Start()

	// Start the transaction pool
	go oc.txPool.Start()

	// Start the node management
	go oc.nodeManagement()

	return nil
}

func (oc *OmniaChain) nodeManagement() {
	for {
		select {
		case newNode := <-oc.network.NewNodeChan:
			oc.addNode(newNode)
		case delNode := <-oc.network.DelNodeChan:
			oc.delNode(delNode)
		}
	}
}

func (oc *OmniaChain) addNode(n *node.Node) {
	oc.nodes = append(oc.nodes, n)
	oc.consensus.AddNode(n)
}

func (oc *OmniaChain) delNode(n *node.Node) {
	for i, node := range oc.nodes {
		if node.ID == n.ID {
			oc.nodes = append(oc.nodes[:i], oc.nodes[i+1:]...)
			oc.consensus.DelNode(n)
			return
		}
	}
}

func (oc *OmniaChain) GetChain() *chain.Chain {
	return oc.chain
}

func (oc *OmniaChain) GetNetwork() *network.Network {
	return oc.network
}

func (oc *OmniaChain) GetConsensus() *consensus.Consensus {
	return oc.consensus
}

func (oc *OmniaChain) GetTxPool() *transaction.TxPool {
	return oc.txPool
}

func main() {
	oc := NewOmniaChain()
	oc.Start()
}
