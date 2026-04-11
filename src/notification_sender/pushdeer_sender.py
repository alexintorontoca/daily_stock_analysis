import requests
import logging

class PushDeerSender:
    def __init__(self, pushkey):
        self.pushkey = pushkey
        self.url = "https://api2.getpushdeer.com/message/push"

    def send(self, title, content):
        if not self.pushkey:
            logging.error("PushDeer pushkey is missing.")
            return False
        
        data = {
            "pushkey": self.pushkey,
            "text": title,
            "desp": content,
            "type": "markdown"
        }
        try:
            response = requests.post(self.url, data=data)
            result = response.json()
            if result.get("code") == 0:
                logging.info("PushDeer notification sent successfully.")
                return True
            else:
                logging.error(f"PushDeer sending failed: {result}")
                return False
        except Exception as e:
            logging.error(f"Error sending PushDeer notification: {e}")
            return False
