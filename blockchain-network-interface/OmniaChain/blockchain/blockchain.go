package blockchain

import (
	"fmt"
	"sync"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/utils"
)

type Blockchain struct {
	chain  []*block.Block
	currentBlock *block.Block
	mu      sync.RWMutex
}

func NewBlockchain() *Blockchain {
	return &Blockchain{
		chain:  make([]*block.Block, 0),
	}
}

func (bc *Blockchain) AddBlock(block *block.Block) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	if len(bc.chain) == 0 {
		bc.chain = append(bc.chain, block)
		bc.currentBlock = block
		return nil
	}

	if block.Header.PreviousHash != bc.currentBlock.Header.Hash {
		return fmt.Errorf("invalid block: previous hash mismatch")
	}

	bc.chain = append(bc.chain, block)
	bc.currentBlock = block
	return nil
}

func (bc *Blockchain) GetChain() []*block.Block {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	return bc.chain
}

func (bc *Blockchain) GetCurrentBlock() *block.Block {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	return bc.currentBlock
}
