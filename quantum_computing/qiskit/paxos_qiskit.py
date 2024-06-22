# paxos_qiskit.py
from qiskit import QuantumCircuit, execute

class PaxosQiskit:
    def __init__(self):
        self.circuit = QuantumCircuit(5)

    def run_circuit(self):
        job = execute(self.circuit, backend='ibmq_qasm_simulator')
        result = job.result()
        return result
