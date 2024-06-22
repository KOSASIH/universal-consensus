// blockchain-network-interface/config.go
package blockchain

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type Config struct {
	Hyperledger struct {
		URL      string `json:"url"`
		Username string `json:"username"`
		Password string `json:"password"`
	} `json:"hyperledger"`
	Ethereum struct {
		URL      string `json:"url"`
		Username string `json:"username"`
		Password string `json:"password"`
	} `json:"ethereum"`
	Quorum struct {
		URL      string `json:"url"`
		Username string `json:"username"`
		Password string `json:"password"`
	} `json:"quorum"`
}

func LoadConfig(file string) (*Config, error) {
	var config Config
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}
	return &config, nil
}
