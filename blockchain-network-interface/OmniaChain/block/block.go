package block

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/transaction"
)

type Block struct {
	Header     BlockHeader
	Transactions []*transaction.Transaction
}

type BlockHeader struct {
	Hash         string
	PrevHash     string
	Timestamp    time.Time
	Nonce        uint64
	Difficulty   uint64
	MerkleRoot   string
}

func NewBlock(transactions []*transaction.Transaction, prevHash string) *Block {
	block := &Block{
		Header: BlockHeader{
			Timestamp: time.Now(),
			Nonce:     0,
			Difficulty: 1,
			MerkleRoot: calculateMerkleRoot(transactions),
		},
		Transactions: transactions,
	}

	block.Header.Hash = calculateBlockHash(block)
	block.Header.PrevHash = prevHash

	return block
}

func calculateBlockHash(block *Block) string {
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("%v", block.Header)))
	hash.Write([]byte(fmt.Sprintf("%v", block.Transactions)))
	return fmt.Sprintf("%x", hash.Sum(nil))
}

func calculateMerkleRoot(transactions []*transaction.Transaction) string {
	hash := sha256.New()
	for _, tx := range transactions {
		hash.Write([]byte(tx.ID))
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
}
