package network

import (
	"fmt"
	"net"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/node"
)

type Network struct {
	nodes    []*node.Node
	nodeMutex sync.RWMutex
	newNodeChan chan *node.Node
	delNodeChan chan *node.Node
}

func NewNetwork() *Network {
	return &Network{
		nodes: make([]*node.Node, 0),
		newNodeChan: make(chan *node.Node),
		delNodeChan: make(chan *node.Node),
	}
}

func (n *Network) Start() {
	go n.runNetwork()
}

func (n *Network) runNetwork() {
	for {
		select {
		case newNode := <-n.newNodeChan:
			n.addNode(newNode)
		case delNode := <-n.delNodeChan:
			n.delNode(delNode)
		}
	}
}

func (n *Network) AddNode(newNode *node.Node) {
	n.nodeMutex.Lock()
	defer n.nodeMutex.Unlock()

	n.nodes = append(n.nodes, newNode)
}

func (n *Network) DelNode(delNode *node.Node) {
	n.nodeMutex.Lock()
	defer n.nodeMutex.Unlock()

	for i, node := range n.nodes {
		if node.ID == delNode.ID {
			n.nodes = append(n.nodes[:i], n.nodes[i+1:]...)
			return
		}
	}
}

func (n *Network) GetNodes() []*node.Node {
	n.nodeMutex.RLock()
	defer n.nodeMutex.RUnlock()

	return n.nodes
}
