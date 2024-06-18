import stellar_sdk
from stellar_sdk.config import Config
from stellar_sdk.server import Server
from stellar_sdk.keypair import Keypair
from stellar_sdk.transaction_builder import TransactionBuilder
from stellar_sdk.exceptions import BadRequestError

class StellarInterface:
    def __init__(self, horizon_url='https://horizon.stellar.org'):
        self.server = Server(horizon_url)

    def create_account(self, public_key, funding_account, starting_balance):
        source_account = Keypair.from_public_key(funding_account)
        transaction = (
            TransactionBuilder(
                source_account=source_account,
                network_passphrase=stellar_sdk.Network.PUBLIC
            )
            .append_create_account_op(
                destination=public_key,
                starting_balance=starting_balance
            )
            .build()
        )

        response = self.server.submit_transaction(transaction)
        return response

    def create_asset(self, issuer_account, asset_code, asset_issuer):
        asset = {
            'code': asset_code,
            'issuer': issuer_account,
            'name': asset_code,
            'desc': 'Sample asset created using StellarInterface',
            'domain': 'example.com',
            'url': 'https://example.com/asset',
            'precision': 7,
            'tick_size': '0.0000001'
        }

        response = self.server.assets.create(asset)
        return response

    def submit_transaction(self, transaction):
        response = self.server.submit_transaction(transaction)
        return response

if __name__ == '__main__':
    stellar_interface = StellarInterface()

    # Create a new account with a starting balance of 1000 XLM
    new_account_public_key = 'GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV'
    funding_account = 'GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7'
    starting_balance = '1000'
    response = stellar_interface.create_account(
        new_account_public_key,
        funding_account,
        starting_balance
    )
    print(response)

    # Create a new asset with code "SAMPLE" and issuer "GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7"
    asset_code = 'SAMPLE'
    asset_issuer = 'GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7XV7'
    response = stellar_interface.create_asset(
        asset_issuer,
        asset_code,
        asset_issuer
    )
    print(response)

    # Submit a sample transaction
    transaction = (
        TransactionBuilder(
            source_account=Keypair.from_public_key(funding_account),
            network_passphrase=stellar_sdk.Network.PUBLIC
        )
        .append_payment_op(
            destination=new_account_public_key,
            amount='10'
        )
        .build()
    )

    response = stellar_interface.submit_transaction(transaction)
    print(response)
