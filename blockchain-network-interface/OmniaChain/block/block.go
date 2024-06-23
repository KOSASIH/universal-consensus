package block

import (
	"crypto/sha256"
	"encoding/hex"
	"time"
)

type Block struct {
	Header     BlockHeader
	Transactions []Transaction
}

type BlockHeader struct {
	Hash         string
	PreviousHash string
	Timestamp    time.Time
	Nonce        uint64
	Difficulty   uint64
}

type Transaction struct {
	ID        string
	Timestamp time.Time
	Sender    string
	Recipient string
	Amount    uint64
}

func NewBlock(transactions []Transaction, previousHash string) *Block {
	block := &Block{
		Header: BlockHeader{
			Timestamp: time.Now(),
			PreviousHash: previousHash,
		},
		Transactions: transactions,
	}

	block.Header.Hash = calculateHash(block)
	return block
}

func calculateHash(block *Block) string {
	hash := sha256.New()
	hash.Write([]byte(block.Header.Timestamp.String()))
	hash.Write([]byte(block.Header.PreviousHash))
	hash.Write([]byte(stringifyTransactions(block.Transactions)))
	return hex.EncodeToString(hash.Sum(nil))
}

func stringifyTransactions(transactions []Transaction) string {
	var txString string
	for _, tx := range transactions {
		txString += tx.ID + tx.Sender + tx.Recipient + string(tx.Amount)
	}
	return txString
}
