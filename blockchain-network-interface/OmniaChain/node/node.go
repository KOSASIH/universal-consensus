package node

import (
	"fmt"
	"net"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/network"
)

type Node struct {
	ID        string
Addr      net.Addr
	Network   *network.Network
}

func NewNode(addr net.Addr, network *network.Network) *Node {
	return &Node{
		ID:      fmt.Sprintf("%v", addr),
		Addr:    addr,
		Network: network,
	}
}
