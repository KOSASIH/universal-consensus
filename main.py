# main.py
import os
import sys
import argparse
from cirq import Circuit, Simulator
from paxos_cirq import PaxosCirq
from raft_cirq import RaftCirq
from pbft_cirq import PBFTCirq

def main():
    parser = argparse.ArgumentParser(description='Universal Consensus')
    parser.add_argument('--algorithm', type=str, required=True, help='Consensus algorithm (paxos, raft, pbft)')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--num_requests', type=int, required=True, help='Number of requests')
    parser.add_argument('--num_replicas', type=int, required=True, help='Number of replicas')
    args = parser.parse_args()

    if args.algorithm == 'paxos':
        consensus = PaxosCirq(args.num_nodes, args.num_requests)
    elif args.algorithm == 'raft':
        consensus = RaftCirq(args.num_nodes, args.num_requests)
    elif args.algorithm == 'pbft':
        consensus = PBFTCirq(args.num_nodes, args.num_requests, args.num_replicas)
    else:
        print('Invalid algorithm')
        sys.exit(1)

    result = consensus.consensus()
    print(result)

if __name__ == '__main__':
    main()
