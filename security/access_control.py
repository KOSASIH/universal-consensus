import os
import json

class AccessControl:
    def __init__(self, config_file='access_control_config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def get_permissions(self, user_id):
        if user_id in self.config['users']:
            return self.config['users'][user_id]['permissions']
        return []

    def check_permission(self, user_id, permission):
        permissions = self.get_permissions(user_id)
        return permission in permissions

    def add_permission(self, user_id, permission):
        if user_id not in self.config['users']:
            self.config['users'][user_id] = {'permissions': []}
        self.config['users'][user_id]['permissions'].append(permission)
        self.save_config()

    def remove_permission(self, user_id, permission):
        if user_id in self.config['users']:
            self.config['users'][user_id]['permissions'].remove(permission)
            self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)

# Example usage:
access_control = AccessControl()
user_id = 'admin'
permission = 'read_write'
if access_control.check_permission(user_id, permission):
    print(f"User {user_id} has permission {permission}")
else:
    print(f"User {user_id} does not have permission {permission}")
    access_control.add_permission(user_id, permission)
    print(f"Added permission {permission} to user {user_id}")
