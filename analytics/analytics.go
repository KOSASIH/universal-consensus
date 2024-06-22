package analytics

import (
	"fmt"
	"time"

	"github.com/KOSASIH/universal-consensus/config"
)

type Analytics struct {
	Interval time.Duration
}

func (a *Analytics) Start() {
	fmt.Println("Analytics started")
}

func (a *Analytics) CollectData() {
	fmt.Println("Collecting data...")
}

func (a *Analytics) ProcessData() {
	fmt.Println("Processing data...")
}

func (a *Analytics) Visualize() {
	fmt.Println("Visualizing data...")
}

func NewAnalytics() *Analytics {
	return &Analytics{Interval: time.Duration(config.GetConfig().Analytics.Interval) * time.Second}
}
