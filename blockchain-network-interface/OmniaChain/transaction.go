package transaction

import (
	"crypto/ecdsa"
	"encoding/json"
	"fmt"
	"math/big"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
)

type Transaction struct {
	ID        string
	From      string
	To        string
	Value     *big.Int
	Timestamp time.Time
	Signature []byte
}

func NewTransaction(from, to string, value *big.Int, privKey *ecdsa.PrivateKey) *Transaction {
	tx := &Transaction{
		ID:      fmt.Sprintf("%v", time.Now().UnixNano()),
		From:    from,
		To:      to,
		Value:   value,
		Timestamp: time.Now(),
	}

	signature, err := ecdsa.Sign(nil, privKey, tx.Hash())
	if err != nil {
		fmt.Println(err)
		return nil
	}
	tx.Signature = signature

	return tx
}

func (tx *Transaction) Hash() []byte {
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("%v", tx)))
	return hash.Sum(nil)
}
