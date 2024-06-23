package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateRandomHash(t *testing.T) {
	hash := GenerateRandomHash()
	assert.NotEmpty(t, hash)
	assert.Equal(t, 64, len(hash))
}

func TestGenerateRandomNonce(t *testing.T) {
	nonce := GenerateRandomNonce()
	assert.NotZero(t, nonce)
}

func TestGenerateRandomDifficulty(t *testing.T) {
	difficulty := GenerateRandomDifficulty()
	assert.NotZero(t, difficulty)
	assert.Equal(t, 8, len(fmt.Sprintf("%d", difficulty)))
}
