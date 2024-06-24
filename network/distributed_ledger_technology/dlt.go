// dlt.go
package main

import (
	"fmt"
	"github.com/hyperledger/fabric-sdk-go/pkg/core/config"
	"github.com/hyperledger/fabric-sdk-go/pkg/fabsdk"
)

type DLT struct {
	sdk *fabsdk.FabricSDK
}

func NewDLT() (*DLT, error) {
	sdk, err := fabsdk.New(config.FromFile("config.yaml"))
	if err != nil {
		return nil, err
	}
	return &DLT{sdk: sdk}, nil
}

func (d *DLT) CreateChannel(channelID string) error {
	// Create a new channel
	channelConfig := `{
		"channel_id": "` + channelID + `",
		"orderer": {
			"orderer.example.com": {
				"url": "grpc://orderer.example.com:7050"
			}
		},
		"peers": {
			"peer0.org1.example.com": {
				"url": "grpc://peer0.org1.example.com:7051"
			},
			"peer0.org2.example.com": {
				"url": "grpc://peer0.org2.example.com:7051"
			}
		}
	}`
	channelConfigJSON := []byte(channelConfig)
	return d.sdk.ChannelService().CreateChannel(channelConfigJSON)
}

func (d *DLT) QueryChaincode(chaincodeID string, args []string) ([]byte, error) {
	// Query a chaincode
	return d.sdk.ChaincodeService().Query(chaincodeID, args)
}
