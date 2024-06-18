import unittest
from stellar import StellarInterface
import json

class TestStellarInterface(unittest.TestCase):
    def setUp(self):
        with open('stellar_config.json') as f:
            self.config = json.load(f)
        self.stellar_interface = StellarInterface(self.config['network']['horizon_url'])

    def test_create_account(self):
        funding_account = Keypair.from_secret(self.config['accounts']['funding_account']['secret_key'])
        new_account_public_key = 'GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV'
        starting_balance = '1000'
        response = self.stellar_interface.create_account(
            new_account_public_key,
            funding_account,
            starting_balance
        )
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    def test_create_asset(self):
        issuer_account = Keypair.from_secret(self.config['accounts']['issuer_account']['secret_key'])
        asset_code = 'SAMPLE'
        asset_issuer = issuer_account.public_key
        response = self.stellar_interface.create_asset(
            asset_issuer,
            asset_code,
            asset_issuer
        )
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    def test_submit_transaction(self):
        funding_account = Keypair.from_secret(self.config['accounts']['funding_account']['secret_key'])
        destination_account = Keypair.from_public_key('GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV')
        amount = '10'
        transaction = (
            TransactionBuilder(
                source_account=funding_account,
                network_passphrase=self.config['network']['network_passphrase']
            )
           .append_payment_op(
                destination=destination_account.public_key,
                amount=amount
            )
           .build()
        )
        response = self.stellar_interface.submit_transaction(transaction)
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    def test_get_account_balance(self):
        account_public_key = 'GC2QQ32PJ37XV7XV7XV7XV7XV7XV7XV7XV7XV7XV'
        response = self.stellar_interface.get_account_balance(account_public_key)
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    def test_get_asset_info(self):
        asset_code = 'SAMPLE'
        asset_issuer = 'GDW6AUTBXTOF7SGBR5CJIRKXLK5T54UNOTJTUW2OEZVYRRTPIRHZPX'
        response = self.stellar_interface.get_asset_info(asset_code, asset_issuer)
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
