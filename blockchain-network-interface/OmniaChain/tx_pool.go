package transaction

import (
	"fmt"
	"sync"

	"github.com/KOSASIH/universal-consensus/blockchain-network-interface/OmniaChain/block"
)

type TxPool struct {
	txs    []*Transaction
	txMutex sync.RWMutex
}

func NewTxPool() *TxPool {
	return &TxPool{
		txs: make([]*Transaction, 0),
	}
}

func (tp *TxPool) AddTx(tx *Transaction) {
	tp.txMutex.Lock()
	defer tp.txMutex.Unlock()

	tp.txs = append(tp.txs, tx)
}

func (tp *TxPool) GetTxByID(id string) (*Transaction, error) {
	tp.txMutex.RLock()
	defer tp.txMutex.RUnlock()

	for _, tx := range tp.txs {
		if tx.ID == id {
			return tx, nil
		}
	}
	return nil, fmt.Errorf("transaction not found")
}

func (tp *TxPool) GetTxs() []*Transaction {
	tp.txMutex.RLock()
	defer tp.txMutex.RUnlock()

	return tp.txs
}
