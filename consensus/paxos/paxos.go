package paxos

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/KOSASIH/universal-consensus/config"
)

type Paxos struct {
	Proposer  int
	Acceptor  int
	Learner   int
	Proposal  string
	Accepted  bool
}

func (p *Paxos) Propose(proposal string) {
	p.Proposal = proposal
	p.Proposer = rand.Intn(config.GetConfig().Network.Nodes)
}

func (p *Paxos) Prepare() {
	fmt.Println("Prepare phase started")
}

func (p *Paxos) Accept() {
	p.Accepted = true
	fmt.Println("Accept phase completed")
}

func (p *Paxos) Learn() {
	fmt.Println("Learn phase started")
}

func NewPaxos() *Paxos {
	return &Paxos{}
}
