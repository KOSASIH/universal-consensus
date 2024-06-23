package consensus

import (
	"fmt"
	"sync"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/node"
)

type Consensus struct {
	nodes    []*node.Node
	nodeMutex sync.RWMutex
}

func NewConsensus() *Consensus {
	return &Consensus{
		nodes: make([]*node.Node, 0),
	}
}

func (c *Consensus) AddNode(n *node.Node) {
	c.nodeMutex.Lock()
	defer c.nodeMutex.Unlock()

	c.nodes = append(c.nodes, n)
}

func (c *Consensus) DelNode(n *node.Node) {
	c.nodeMutex.Lock()
	defer c.nodeMutex.Unlock()

	for i, node := range c.nodes {
		if node.ID == n.ID {
			c.nodes = append(c.nodes[:i], c.nodes[i+1:]...)
			return
		}
	}
}

func (c *Consensus) Start() {
	go c.runConsensus()
}

func (c *Consensus) runConsensus() {
	for {
		select {
		case <-time.After(10 * time.Second):
			c.runRound()
		}
	}
}

func (c *Consensus) runRound() {
	// Implement the consensus algorithm logic here
	fmt.Println("Running consensus round...")
}
