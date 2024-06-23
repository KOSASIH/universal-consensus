package chain

import (
	"fmt"
	"sync"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
)

type Chain struct {
	blocks    []*block.Block
	blockMutex sync.RWMutex
}

func NewChain() *Chain {
	return &Chain{
		blocks: make([]*block.Block, 0),
	}
}

func (c *Chain) AddBlock(b *block.Block) error {
	c.blockMutex.Lock()
	defer c.blockMutex.Unlock()

	c.blocks = append(c.blocks, b)
	return nil
}

func (c *Chain) GetBlockByHash(hash string) (*block.Block, error) {
	c.blockMutex.RLock()
	defer c.blockMutex.RUnlock()

	for _, b := range c.blocks {
		if b.Header.Hash == hash {
			return b, nil
		}
	}
	return nil, fmt.Errorf("block not found")
}

func (c *Chain) GetBlockByIndex(index uint64) (*block.Block, error) {
	c.blockMutex.RLock()
	defer c.blockMutex.RUnlock()

	if index >= uint64(len(c.blocks)) {
		return nil, fmt.Errorf("block not found")
	}
	return c.blocks[index], nil
}

func (c *Chain) GetLatestBlock() (*block.Block, error) {
	c.blockMutex.RLock()
	defer c.blockMutex.RUnlock()

	if len(c.blocks) == 0 {
		return nil, fmt.Errorf("no blocks in chain")
	}
	return c.blocks[len(c.blocks)-1], nil
}
