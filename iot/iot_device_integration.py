import requests

class IoTDeviceIntegration:
    def __init__(self, device_url):
        self.device_url = device_url

    def send_data(self, data):
        requests.post(self.device_url, json=data)

    def receive_data(self):
        response = requests.get(self.device_url)
        return response.json()
