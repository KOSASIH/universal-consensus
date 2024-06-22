package network

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/KOSASIH/universal-consensus/config"
)

type Node struct {
	ID       int
	Neighbors []int
}

func (n *Node) Start() {
	fmt.Println("Node", n.ID, "started")
}

func (n *Node) Connect(neighbors []int) {
	n.Neighbors = neighbors
}

func (n *Node) Send(message string) {
	fmt.Println("Node", n.ID, "sent message:", message)
}

func NewNetwork() *Network {
	nodes := make([]*Node, config.GetConfig().Network.Nodes)
	for i := range nodes {
		nodes[i] = &Node{ID: i}
	}
	return &Network{Nodes: nodes}
}

type Network struct {
	Nodes []*Node
}

func (n *Network) Start() {
	for _, node := range n.Nodes {
		node.Start()
	}
}
