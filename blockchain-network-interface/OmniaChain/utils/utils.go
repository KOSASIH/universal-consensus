package utils

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math/big"
)

func GenerateRandomHash() string {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		fmt.Println("Error generating random hash:", err)
		return ""
	}

	return hex.EncodeToString(b)
}

func GenerateRandomNonce() uint64 {
	b := make([]byte, 8)
	_, err := rand.Read(b)
	if err != nil {
		fmt.Println("Error generating random nonce:", err)
		return 0
	}

	return binary.LittleEndian.Uint64(b)
}

func GenerateRandomDifficulty() uint64 {
	difficulty := big.NewInt(1)
	difficulty.Lsh(difficulty, 256)
	difficulty.Rsh(difficulty, 256-8)

	return uint64(difficulty.Int64())
}
