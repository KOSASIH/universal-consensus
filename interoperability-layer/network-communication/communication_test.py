# network-communication/communication_test.py
import unittest
from communication import Communication

class TestCommunication(unittest.TestCase):
    def setUp(self):
        self.host = "localhost"
        self.port = 12345
        self.communication = Communication(self.host, self.port)

    def tearDown(self):
        self.communication.stop_listening()

    def test_send_and_receive(self):
        message = "Hello, world!"
        self.communication.send(message)
        received_message = self.communication.receive()
        self.assertEqual(message, received_message)

    def test_start_and_stop_listening(self):
        def callback(message):
            self.received_messages.append(message)

        self.received_messages = []
        self.communication.start_listening(callback)
        self.communication.send("Hello, world!")
        self.communication.send("Goodbye, world!")
        time.sleep(0.1)
        self.assertEqual(["Hello, world!", "Goodbye, world!"], self.received_messages)

if __name__ == "__main__":
    unittest.main()
