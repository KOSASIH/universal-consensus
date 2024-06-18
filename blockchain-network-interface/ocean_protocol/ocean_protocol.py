import os
from ocean_protocol_sdk import Ocean

class OceanProtocol:
    def __init__(self, provider_url: str, account: str, private_key: str):
        self.ocean = Ocean(provider_url)
        self.account = account
        self.private_key = private_key

    def create_wallet(self):
        """Create a new wallet."""
        return self.ocean.create_wallet(self.private_key)

    def get_account_balance(self):
        """Get the account balance."""
        return self.ocean.get_account_balance(self.account)

    def create_datatoken(self, metadata_uri: str):
        """Create a new datatoken."""
        return self.ocean.create_datatoken(self.account, metadata_uri)

    def list_assets(self):
        """List available assets."""
        return self.ocean.list_assets()

    def get_asset_details(self, asset_id: str):
        """Get asset details."""
        return self.ocean.get_asset_details(asset_id)

    def create_order(self, asset_id: str, amount: str, price: str):
        """Create a new order."""
        return self.ocean.create_order(self.account, asset_id, amount, price)

    def get_orders(self, asset_id: str):
        """Get orders for an asset."""
        return self.ocean.get_orders(asset_id)

    def execute_order(self, order_id: str):
        """Execute an order."""
        return self.ocean.execute_order(self.account, order_id)

    def get_transactions(self):
        """Get account transactions."""
        return self.ocean.get_transactions(self.account)

if __name__ == "__main__":
    # Initialize the Ocean Protocol instance
    ocean_protocol = OceanProtocol(
        provider_url="https://kovan.oceanprotocol.com",
        account="<YOUR_ACCOUNT_ADDRESS>",
        private_key="<YOUR_PRIVATE_KEY>",
    )

    # Example usage
    print("Account balance:", ocean_protocol.get_account_balance())
    print("Listing assets...")
    assets = ocean_protocol.list_assets()
    for asset in assets:
        print(f"Asset ID: {asset['id']}, Name: {asset['name']}")
