// chaincode/chaincode.go
package chaincode

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
)

type Chaincode struct {
}

func (c *Chaincode) Init(stub shim.ChaincodeStubInterface) []byte {
	fmt.Println("Chaincode initialized")
	return nil
}

func (c *Chaincode) Invoke(stub shim.ChaincodeStubInterface) ([]byte, error) {
	fmt.Println("Chaincode invoked")
	return nil, nil
}

func (c *Chaincode) Query(stub shim.ChaincodeStubInterface) ([]byte, error) {
	fmt.Println("Chaincode queried")
	return nil, nil
}

type Args struct {
	Function string `json:"function"`
	Args     []string `json:"args"`
}

func (c *Chaincode) InvokeStub(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	var a Args
	err := json.Unmarshal([]byte(args[0]), &a)
	if err!= nil {
		return nil, err
	}

	switch a.Function {
	case "put":
		return c.put(stub, a.Args)
	case "get":
		return c.get(stub, a.Args)
	default:
		return nil, fmt.Errorf("unknown function %s", a.Function)
	}
}

func (c *Chaincode) put(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	key := args[0]
	value := args[1]
	err := stub.PutState(key, []byte(value))
	if err!= nil {
		return nil, err
	}
	return []byte("put successful"), nil
}

func (c *Chaincode) get(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	key := args[0]
	value, err := stub.GetState(key)
	if err!= nil {
		return nil, err
	}
	return value, nil
}
