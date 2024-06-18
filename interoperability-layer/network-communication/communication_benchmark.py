# network-communication/communication_benchmark.py
import timeit
from communication import Communication

def benchmark_send_and_receive():
    communication = Communication("localhost", 12345)
    message = "Hello, world!"
    communication.send(message)
    received_message = communication.receive()
    return timeit.timeit(lambda: communication.send(message), number=1000)

if __name__ == "__main__":
    print("Send and receive benchmark:", benchmark_send_and_receive())
