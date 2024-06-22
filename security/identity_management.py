import hashlib

class IdentityManagement:
    def __init__(self):
        self.identity_map = {}

    def create_identity(self, user_id, password):
        salt = os.urandom(16)
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        self.identity_map[user_id] = {'salt': salt, 'hashed_password': hashed_password}

    def authenticate(self, user_id, password):
        if user_id in self.identity_map:
            salt = self.identity_map[user_id]['salt']
            hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            if hashed_password == self.identity_map[user_id]['hashed_password']:
                return True
        return False
