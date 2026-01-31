import { ethers } from 'ethers'

/**
 * Donate ETH to a disaster relief vault
 * @param vaultAddress The address of the vault contract
 * @param amountInEth The amount to donate in ETH (as a string, e.g., "0.1")
 * @param signer The ethers signer from the connected wallet
 * @returns Transaction hash
 */
export async function donateToVault(
  vaultAddress: string,
  amountInEth: string,
  signer: ethers.Signer
): Promise<string> {
  try {
    // Convert ETH amount to Wei
    const amountInWei = ethers.parseEther(amountInEth)

    // Send ETH directly to the vault address
    // The vault's receive() function will handle it
    const tx = await signer.sendTransaction({
      to: vaultAddress,
      value: amountInWei,
    })

    // Wait for transaction to be mined
    await tx.wait()

    return tx.hash
  } catch (error: any) {
    console.error('Donation error:', error)
    throw new Error(error.message || 'Failed to process donation')
  }
}

/**
 * Get the provider for Sepolia testnet
 */
export function getProvider(): ethers.JsonRpcProvider {
  return new ethers.JsonRpcProvider('https://sepolia.infura.io/v3/b4880ead6a9a4f77a6de39dec6f3d0d0')
}

/**
 * Create ethers signer from Privy wallet
 * @param wallet The Privy wallet object
 * @returns Promise<ethers.Signer>
 */
export async function createSignerFromPrivyWallet(wallet: any): Promise<ethers.Signer> {
  try {
    // Privy wallet has a getEthereumProvider method that returns an EIP-1193 provider
    const ethereumProvider = await wallet.getEthereumProvider()
    
    if (!ethereumProvider) {
      throw new Error('Wallet provider not available')
    }
    
    const provider = new ethers.BrowserProvider(ethereumProvider)
    const signer = await provider.getSigner()
    return signer
  } catch (error: any) {
    throw new Error(`Failed to create signer: ${error.message}`)
  }
}
