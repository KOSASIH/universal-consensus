# network-communication/communication.py
import socket
import threading

class Communication:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def send(self, message: str):
        self.socket.sendall(message.encode("utf-8"))

    def receive(self) -> str:
        return self.socket.recv(1024).decode("utf-8")

    def start_listening(self, callback):
        def listener():
            while True:
                message = self.receive()
                callback(message)

        thread = threading.Thread(target=listener)
        thread.start()

    def stop_listening(self):
        self.socket.close()
