// chaincode/chaincode_test.go
package chaincode

import (
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
)

func TestChaincode_Init(t *testing.T) {
	stub := shim.NewMockStub("chaincode", new(Chaincode))
	if stub == nil {
		t.Fatal("Failed to create mock stub")
	}

	_, err := stub.MockInit("init", [][]byte{[]byte("init")})
	if err!= nil {
		t.Fatal(err)
	}
}

func TestChaincode_Invoke(t *testing.T) {
	stub := shim.NewMockStub("chaincode", new(Chaincode))
	if stub == nil {
		t.Fatal("Failed to create mock stub")
	}

	args := `{"function": "put", "args": ["key", "value"]}`
	_, err := stub.MockInvoke("invoke", [][]byte{[]byte(args)})
	if err!= nil {
		t.Fatal(err)
	}
}

func TestChaincode_Query(t *testing.T) {
	stub := shim.NewMockStub("chaincode", new(Chaincode))
	if stub == nil {
		t.Fatal("Failed to create mock stub")
	}

	args := `{"function": "get", "args": ["key"]}`
	_, err := stub.MockInvoke("query", [][]byte{[]byte(args)})
	if err!= nil {
		t.Fatal(err)
	}
}
