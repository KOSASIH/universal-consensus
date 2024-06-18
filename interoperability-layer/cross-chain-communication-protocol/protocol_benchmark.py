# cross-chain-communication-protocol/protocol_benchmark.py
import timeit
from protocol import CrossChainCommunicationProtocol

def benchmark_sign_message():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    chain_id = "test_chain"
    protocol = CrossChainCommunicationProtocol(private_key, chain_id)
    message = "Hello, world!"
    return timeit.timeit(lambda: protocol.sign_message(message), number=1000)

def benchmark_verify_signature():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    chain_id = "test_chain"
    protocol = CrossChainCommunicationProtocol(private_key, chain_id)
    message = "Hello, world!"
    signature = protocol.sign_message(message)
    return timeit.timeit(lambda: protocol.verify_signature(message, signature), number=1000)

if __name__ == "__main__":
    print("Sign message benchmark:", benchmark_sign_message())
    print("Verify signature benchmark:", benchmark_verify_signature())
