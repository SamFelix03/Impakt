"""
Simple script to donate funds to a specific vault
Requires: pip install web3 python-dotenv
"""

from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
RPC_URL = 'https://ethereum-sepolia-rpc.publicnode.com'
PRIVATE_KEY = os.getenv('PRIVATE_KEY')

# Hardcoded values
VAULT_ADDRESS = "0x158bf54057cC995B5cddDd2757761043209Fa308"
DONATION_AMOUNT = 0.0004  # ETH


def donate_to_vault():
    """Send donation to the hardcoded vault address"""
    try:
        # Connect to Sepolia testnet
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        
        # Check connection
        if not w3.is_connected():
            raise Exception("Failed to connect to Ethereum node")
        
        print(f"✅ Connected to Ethereum Sepolia")
        print(f"Chain ID: {w3.eth.chain_id}\n")
        
        # Setup account from private key
        if not PRIVATE_KEY:
            raise Exception("PRIVATE_KEY not found in .env file")
        
        # Remove '0x' prefix if present
        private_key = PRIVATE_KEY if not PRIVATE_KEY.startswith('0x') else PRIVATE_KEY[2:]
        account = Account.from_key(private_key)
        
        print(f"Donor Wallet: {account.address}")
        
        # Check wallet balance
        balance = w3.eth.get_balance(account.address)
        print(f"Wallet Balance: {w3.from_wei(balance, 'ether')} ETH\n")
        
        # Convert vault address to checksum format
        vault_checksum = Web3.to_checksum_address(VAULT_ADDRESS)
        
        print("=== DONATING TO VAULT ===")
        print(f"Vault Address: {vault_checksum}")
        print(f"Donation Amount: {DONATION_AMOUNT} ETH")
        
        # Convert ETH to Wei
        donation_wei = w3.to_wei(DONATION_AMOUNT, 'ether')
        
        # Check if wallet has enough balance
        if balance < donation_wei:
            raise Exception(f"Insufficient balance. Need {DONATION_AMOUNT} ETH but only have {w3.from_wei(balance, 'ether')} ETH")
        
        # Get nonce
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Estimate gas for the transaction
        print("\nEstimating gas...")
        gas_estimate = w3.eth.estimate_gas({
            'from': account.address,
            'to': vault_checksum,
            'value': donation_wei
        })
        
        # Add 20% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.2)
        
        # Get current gas price and calculate fees
        base_fee = w3.eth.gas_price
        max_priority_fee = w3.to_wei(2, 'gwei')
        max_fee = base_fee + max_priority_fee
        
        print(f"Estimated gas: {gas_estimate}")
        print(f"Gas limit (with buffer): {gas_limit}")
        print(f"Base fee: {w3.from_wei(base_fee, 'gwei')} gwei")
        print(f"Max priority fee: {w3.from_wei(max_priority_fee, 'gwei')} gwei")
        print(f"Max fee: {w3.from_wei(max_fee, 'gwei')} gwei")
        
        # Calculate total cost
        total_cost = donation_wei + (gas_limit * max_fee)
        print(f"Total transaction cost (including gas): {w3.from_wei(total_cost, 'ether')} ETH\n")
        
        # Build transaction
        tx = {
            'from': account.address,
            'to': vault_checksum,
            'value': donation_wei,
            'nonce': nonce,
            'gas': gas_limit,
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': max_priority_fee,
            'chainId': w3.eth.chain_id
        }
        
        # Sign transaction
        print("Signing transaction...")
        signed_tx = account.sign_transaction(tx)
        
        # Send transaction
        print("Sending transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        print(f"✅ Transaction submitted!")
        print(f"Transaction hash: {tx_hash.hex()}")
        print(f"View on Etherscan: https://sepolia.etherscan.io/tx/{tx_hash.hex()}")
        
        # Wait for confirmation
        print("\nWaiting for confirmation...")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        print(f"\n🎉 DONATION SUCCESSFUL!")
        print(f"Block number: {tx_receipt['blockNumber']}")
        print(f"Gas used: {tx_receipt['gasUsed']}")
        print(f"Status: {'Success' if tx_receipt['status'] == 1 else 'Failed'}")
        
        # Show new wallet balance
        new_balance = w3.eth.get_balance(account.address)
        print(f"\nNew wallet balance: {w3.from_wei(new_balance, 'ether')} ETH")
        print(f"Amount donated: {DONATION_AMOUNT} ETH")
        
        return tx_receipt
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("DISASTER RELIEF VAULT DONATION SCRIPT")
    print("=" * 60)
    print()
    
    donate_to_vault()