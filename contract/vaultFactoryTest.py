"""
Disaster Relief Vault Interaction Script
Requires: pip install web3 python-dotenv
"""

from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
RPC_URL = 'https://ethereum-sepolia-rpc.publicnode.com'
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
FACTORY_ADDRESS = os.getenv('FACTORY_ADDRESS')  # Add your deployed factory address to .env

# Contract ABIs
FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "_disasterName", "type": "string"},
            {"internalType": "uint256", "name": "_reliefAmount", "type": "uint256"}
        ],
        "name": "createVault",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getVaultCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getVaultByIndex",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "string", "name": "_disasterName", "type": "string"}],
        "name": "getVaultByDisaster",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAllVaultBalances",
        "outputs": [
            {"internalType": "address[]", "name": "", "type": "address[]"},
            {"internalType": "uint256[]", "name": "", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAllVaultDetails",
        "outputs": [
            {"internalType": "address[]", "name": "addresses", "type": "address[]"},
            {"internalType": "string[]", "name": "disasterNames", "type": "string[]"},
            {"internalType": "uint256[]", "name": "reliefAmounts", "type": "uint256[]"},
            {"internalType": "uint256[]", "name": "currentBalances", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

VAULT_ABI = [
    {
        "inputs": [],
        "name": "disasterName",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "reliefAmount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalReceived",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalWithdrawn",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "creator",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getVaultInfo",
        "outputs": [
            {"internalType": "string", "name": "", "type": "string"},
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_to", "type": "address"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"}
        ],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_to", "type": "address"}
        ],
        "name": "withdrawAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


def setup_web3():
    """Initialize Web3 connection and wallet"""
    # Connect to Sepolia testnet
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    
    # Check connection
    if not w3.is_connected():
        raise Exception("Failed to connect to Ethereum node")
    
    print(f"✅ Connected to Ethereum Sepolia")
    print(f"Chain ID: {w3.eth.chain_id}")
    
    # Setup account from private key
    if not PRIVATE_KEY:
        raise Exception("PRIVATE_KEY not found in .env file")
    
    # Remove '0x' prefix if present
    private_key = PRIVATE_KEY if not PRIVATE_KEY.startswith('0x') else PRIVATE_KEY[2:]
    account = Account.from_key(private_key)
    
    print(f"Wallet Address: {account.address}")
    
    # Check balance
    balance = w3.eth.get_balance(account.address)
    print(f"Wallet Balance: {w3.from_wei(balance, 'ether')} ETH\n")
    
    return w3, account


def create_disaster_vault(w3, account, factory_contract, disaster_name, relief_amount_eth):
    """Create a new disaster relief vault"""
    print("=== CREATING NEW DISASTER VAULT ===")
    print(f"Disaster Name: {disaster_name}")
    print(f"Target Relief Amount: {relief_amount_eth} ETH")
    
    # Convert ETH to Wei
    relief_amount_wei = w3.to_wei(relief_amount_eth, 'ether')
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Estimate gas for the transaction
    gas_estimate = factory_contract.functions.createVault(
        disaster_name,
        relief_amount_wei
    ).estimate_gas({'from': account.address})
    
    # Add 20% buffer to gas estimate for safety
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
    
    tx = factory_contract.functions.createVault(
        disaster_name,
        relief_amount_wei
    ).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': gas_limit,
        'maxFeePerGas': max_fee,
        'maxPriorityFeePerGas': max_priority_fee,
        'chainId': w3.eth.chain_id  # EIP-155 replay protection
    })
    
    # Sign and send transaction
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"Transaction submitted: {tx_hash.hex()}")
    
    # Wait for confirmation
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Transaction confirmed in block: {tx_receipt['blockNumber']}")
    print(f"Gas used: {tx_receipt['gasUsed']}\n")
    
    return tx_receipt


def get_vault_address(factory_contract, disaster_name):
    """Get vault address by disaster name"""
    print("=== GETTING VAULT ADDRESS ===")
    vault_address = factory_contract.functions.getVaultByDisaster(disaster_name).call()
    print(f"Vault Address: {vault_address}\n")
    return vault_address


def send_donation(w3, account, vault_address, donation_amount_eth):
    """Send ETH donation to vault"""
    print("=== SENDING DONATION ===")
    print(f"Sending {donation_amount_eth} ETH to vault...")
    
    donation_wei = w3.to_wei(donation_amount_eth, 'ether')
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Estimate gas for the transaction
    gas_estimate = w3.eth.estimate_gas({
        'from': account.address,
        'to': vault_address,
        'value': donation_wei
    })
    
    # Add 20% buffer to gas estimate for safety
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
    
    tx = {
        'from': account.address,
        'to': vault_address,
        'value': donation_wei,
        'nonce': nonce,
        'gas': gas_limit,
        'maxFeePerGas': max_fee,
        'maxPriorityFeePerGas': max_priority_fee,
        'chainId': w3.eth.chain_id  # EIP-155 replay protection
    }
    
    # Sign and send
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"Donation transaction: {tx_hash.hex()}")
    
    # Wait for confirmation
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Donation confirmed!")
    print(f"Gas used: {tx_receipt['gasUsed']}\n")
    
    return tx_receipt


def track_all_vaults(w3, factory_contract):
    """Track progress of all vaults"""
    print("=== TRACKING ALL VAULTS ===")
    
    vault_count = factory_contract.functions.getVaultCount().call()
    print(f"Total vaults created: {vault_count}")
    
    if vault_count == 0:
        print("No vaults created yet.")
        return
    
    # Get all vault details
    addresses, disaster_names, relief_amounts, current_balances = \
        factory_contract.functions.getAllVaultDetails().call()
    
    print("\n--- VAULT PROGRESS REPORT ---\n")
    
    total_funds = 0
    
    for i in range(len(addresses)):
        progress = (current_balances[i] / relief_amounts[i]) * 100 if relief_amounts[i] > 0 else 0
        
        print(f"Vault #{i + 1}:")
        print(f"  Address: {addresses[i]}")
        print(f"  Disaster: {disaster_names[i]}")
        print(f"  Target: {w3.from_wei(relief_amounts[i], 'ether')} ETH")
        print(f"  Current: {w3.from_wei(current_balances[i], 'ether')} ETH")
        print(f"  Progress: {progress:.4f}%")
        print(f"  Status: {'✅ GOAL REACHED' if progress >= 100 else '🔄 In Progress'}")
        print()
        
        total_funds += current_balances[i]
    
    print(f"💰 Total funds across all vaults: {w3.from_wei(total_funds, 'ether')} ETH\n")


def withdraw_from_vault(w3, account, vault_address, recipient_address, amount_eth):
    """Withdraw funds from vault to a specific address"""
    print("=== WITHDRAWING FROM VAULT ===")
    print(f"Withdrawing {amount_eth} ETH to {recipient_address}...")
    
    # Connect to vault contract
    vault_contract = w3.eth.contract(address=vault_address, abi=VAULT_ABI)
    
    # Convert ETH to Wei
    amount_wei = w3.to_wei(amount_eth, 'ether')
    
    # Check current balance
    current_balance = vault_contract.functions.getBalance().call()
    print(f"Current vault balance: {w3.from_wei(current_balance, 'ether')} ETH")
    
    if amount_wei > current_balance:
        raise Exception(f"Insufficient balance. Requested: {amount_eth} ETH, Available: {w3.from_wei(current_balance, 'ether')} ETH")
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Convert recipient address to checksum format
    recipient_checksum = Web3.to_checksum_address(recipient_address)
    
    # Estimate gas for the transaction
    gas_estimate = vault_contract.functions.withdraw(
        recipient_checksum,
        amount_wei
    ).estimate_gas({'from': account.address})
    
    # Add 20% buffer to gas estimate for safety
    gas_limit = int(gas_estimate * 1.2)
    
    # Get current gas price and calculate fees
    base_fee = w3.eth.gas_price
    max_priority_fee = w3.to_wei(2, 'gwei')
    max_fee = base_fee + max_priority_fee
    
    print(f"Estimated gas: {gas_estimate}")
    print(f"Gas limit (with buffer): {gas_limit}")
    
    tx = vault_contract.functions.withdraw(
        recipient_checksum,
        amount_wei
    ).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': gas_limit,
        'maxFeePerGas': max_fee,
        'maxPriorityFeePerGas': max_priority_fee,
        'chainId': w3.eth.chain_id
    })
    
    # Sign and send transaction
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"Withdrawal transaction: {tx_hash.hex()}")
    
    # Wait for confirmation
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Withdrawal confirmed!")
    print(f"Gas used: {tx_receipt['gasUsed']}")
    
    # Check new balance
    new_balance = vault_contract.functions.getBalance().call()
    print(f"New vault balance: {w3.from_wei(new_balance, 'ether')} ETH\n")
    
    return tx_receipt


def withdraw_all_from_vault(w3, account, vault_address, recipient_address):
    """Withdraw all funds from vault to a specific address"""
    print("=== WITHDRAWING ALL FUNDS FROM VAULT ===")
    print(f"Withdrawing all funds to {recipient_address}...")
    
    # Connect to vault contract
    vault_contract = w3.eth.contract(address=vault_address, abi=VAULT_ABI)
    
    # Check current balance
    current_balance = vault_contract.functions.getBalance().call()
    print(f"Current vault balance: {w3.from_wei(current_balance, 'ether')} ETH")
    
    if current_balance == 0:
        print("No funds to withdraw.")
        return None
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Convert recipient address to checksum format
    recipient_checksum = Web3.to_checksum_address(recipient_address)
    
    # Estimate gas for the transaction
    gas_estimate = vault_contract.functions.withdrawAll(
        recipient_checksum
    ).estimate_gas({'from': account.address})
    
    # Add 20% buffer to gas estimate for safety
    gas_limit = int(gas_estimate * 1.2)
    
    # Get current gas price and calculate fees
    base_fee = w3.eth.gas_price
    max_priority_fee = w3.to_wei(2, 'gwei')
    max_fee = base_fee + max_priority_fee
    
    print(f"Estimated gas: {gas_estimate}")
    print(f"Gas limit (with buffer): {gas_limit}")
    
    tx = vault_contract.functions.withdrawAll(
        recipient_checksum
    ).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': gas_limit,
        'maxFeePerGas': max_fee,
        'maxPriorityFeePerGas': max_priority_fee,
        'chainId': w3.eth.chain_id
    })
    
    # Sign and send transaction
    signed_tx = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print(f"Withdrawal transaction: {tx_hash.hex()}")
    
    # Wait for confirmation
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ All funds withdrawn successfully!")
    print(f"Gas used: {tx_receipt['gasUsed']}")
    print(f"Amount withdrawn: {w3.from_wei(current_balance, 'ether')} ETH\n")
    
    return tx_receipt


def track_specific_vault(w3, vault_address):
    """Track a specific vault by address"""
    vault_contract = w3.eth.contract(address=vault_address, abi=VAULT_ABI)
    
    name, target, total_received, total_withdrawn, current_balance, creator = vault_contract.functions.getVaultInfo().call()
    
    print("\n--- VAULT DETAILS ---")
    print(f"Address: {vault_address}")
    print(f"Creator: {creator}")
    print(f"Disaster: {name}")
    print(f"Target Amount: {w3.from_wei(target, 'ether')} ETH")
    print(f"Total Received: {w3.from_wei(total_received, 'ether')} ETH")
    print(f"Total Withdrawn: {w3.from_wei(total_withdrawn, 'ether')} ETH")
    print(f"Current Balance: {w3.from_wei(current_balance, 'ether')} ETH")
    
    progress = (current_balance / target * 100) if target > 0 else 0
    print(f"Progress: {progress:.4f}%\n")


def main():
    """Main execution function"""
    try:
        # Check if factory address is set
        if not FACTORY_ADDRESS:
            raise Exception("FACTORY_ADDRESS not found in .env file. Please deploy the contract first and add the address to .env")
        
        # Setup Web3 and account
        w3, account = setup_web3()
        
        # Connect to factory contract
        factory_contract = w3.eth.contract(
            address=Web3.to_checksum_address(FACTORY_ADDRESS),
            abi=FACTORY_ABI
        )
        
        # 1. CREATE A NEW DISASTER VAULT
        disaster_name = "Earthquake Relief 2024"
        relief_amount = 5  # 5 ETH target
        
        create_disaster_vault(w3, account, factory_contract, disaster_name, relief_amount)
        
        # 2. GET THE VAULT ADDRESS
        vault_address = get_vault_address(factory_contract, disaster_name)
        
        # 3. SEND 0.00001 ETH TO THE VAULT
        donation_amount = 0.00001  # ETH
        send_donation(w3, account, vault_address, donation_amount)
        
        # Verify the donation
        vault_contract = w3.eth.contract(address=vault_address, abi=VAULT_ABI)
        vault_balance = vault_contract.functions.getBalance().call()
        print(f"Vault balance after donation: {w3.from_wei(vault_balance, 'ether')} ETH\n")
        
        # 4. WITHDRAW FUNDS FROM VAULT
        recipient_address = "0x5732e1bccAEB161E3B93D126010042B0F1b9CFC9"
        
        # Option A: Withdraw specific amount (half of the donation)
        # withdrawal_amount = donation_amount / 2
        # withdraw_from_vault(w3, account, vault_address, recipient_address, withdrawal_amount)
        
        # Option B: Withdraw all funds
        withdraw_all_from_vault(w3, account, vault_address, recipient_address)
        
        # 5. TRACK PROGRESS OF ALL VAULTS
        track_all_vaults(w3, factory_contract)
        
        # 6. TRACK SPECIFIC VAULT DETAILS
        track_specific_vault(w3, vault_address)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()