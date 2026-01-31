'use client'

import { useState, useEffect, useRef } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2, Wallet, CheckCircle, ExternalLink, AlertCircle, Copy, Check, LogOut, X } from 'lucide-react'
import { donateToVault, createSignerFromPrivyWallet, getProvider } from '@/lib/donation'
import { useAuth } from '@/lib/auth'
import { convertEthToUsd, getEthPriceInUsd } from '@/lib/eth-to-usd'
import { useConnectWallet, useWallets } from '@privy-io/react-auth'
import { truncateAddress } from '@/lib/utils'
import { ethers } from 'ethers'

interface DonationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  vaultAddress: string
  eventId: string
  onDonationSuccess: () => void
}

type Step = 'connect' | 'donate' | 'success'

export function DonationModal({
  open,
  onOpenChange,
  vaultAddress,
  eventId,
  onDonationSuccess,
}: DonationModalProps) {
  const { dbUser } = useAuth()
  const { connectWallet } = useConnectWallet()
  const { wallets } = useWallets()
  const [step, setStep] = useState<Step>('connect')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [amount, setAmount] = useState('')
  const [usdAmount, setUsdAmount] = useState<number | null>(null)
  const [ethPrice, setEthPrice] = useState<number | null>(null)
  const [converting, setConverting] = useState(false)
  const [txHash, setTxHash] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [ethBalance, setEthBalance] = useState<string | null>(null)
  const userNavigatedBack = useRef(false)

  // Reset state when modal opens
  useEffect(() => {
    if (open) {
      setStep('connect')
      setAmount('')
      setUsdAmount(null)
      setTxHash(null)
      setError(null)
      setLoading(false)
      setEthPrice(null) // Reset ETH price when modal closes
      userNavigatedBack.current = false
    } else {
      // Reset all state when modal closes
      setEthPrice(null)
      setUsdAmount(null)
      setAmount('')
    }
  }, [open])

  // Auto-advance to donate step when wallet connects (only on connect step, and not if user navigated back)
  useEffect(() => {
    if (step === 'connect' && !userNavigatedBack.current && wallets && wallets.length > 0 && wallets[0].address) {
      setStep('donate')
      setError(null)
      setLoading(false)
    }
  }, [wallets, step])

  // Fetch ETH balance when wallet is connected (from Sepolia testnet)
  useEffect(() => {
    const fetchBalance = async () => {
      if (wallets && wallets.length > 0 && wallets[0].address && step === 'donate') {
        try {
          const wallet = wallets[0]
          // Use Sepolia provider directly to ensure we get Sepolia balance
          const sepoliaProvider = getProvider()
          const balance = await sepoliaProvider.getBalance(wallet.address)
          const balanceInEth = ethers.formatEther(balance)
          // Format to 4 decimal places
          setEthBalance(parseFloat(balanceInEth).toFixed(4))
        } catch (err) {
          console.error('Failed to fetch ETH balance:', err)
          setEthBalance(null)
        }
      } else {
        setEthBalance(null)
      }
    }

    fetchBalance()
  }, [wallets, step])

  // Fetch ETH price ONLY when modal is open AND user is on donate step
  useEffect(() => {
    if (open && step === 'donate' && !ethPrice) {
      getEthPriceInUsd()
        .then(price => setEthPrice(price))
        .catch(err => console.error('Failed to fetch ETH price:', err))
    }
    
    // Reset ethPrice when modal closes
    if (!open) {
      setEthPrice(null)
    }
  }, [open, step, ethPrice])

  // Convert ETH to USD ONLY when modal is open, on donate step, AND user has entered an amount
  useEffect(() => {
    if (open && step === 'donate' && amount && ethPrice && !isNaN(parseFloat(amount)) && parseFloat(amount) > 0) {
      setConverting(true)
      convertEthToUsd(amount)
        .then(usd => {
          setUsdAmount(usd)
          setConverting(false)
        })
        .catch(err => {
          console.error('Conversion error:', err)
          setUsdAmount(null)
          setConverting(false)
        })
    } else {
      setUsdAmount(null)
      setConverting(false)
    }
  }, [open, step, amount, ethPrice])

  const handleConnectWallet = async () => {
    try {
      setLoading(true)
      setError(null)
      userNavigatedBack.current = false // Reset flag when connecting
      
      // Connect wallet using Privy - this opens Privy's wallet selection modal
      await connectWallet()
      
      // The useEffect above will automatically detect the wallet connection
      // and advance to the donate step
      
    } catch (err: any) {
      // User cancelled or closed the Privy modal
      if (err.code === 'USER_REJECTED' || err.message?.includes('rejected') || err.message?.includes('cancelled') || err.message?.includes('closed')) {
        setError(null) // Clear error for cancellation
        setLoading(false) // Reset loading state
      } else {
        setError(err.message || 'Failed to connect wallet')
        setLoading(false)
      }
    }
  }


  const handleCopyAddress = async () => {
    if (!currentWallet?.address) return
    
    try {
      await navigator.clipboard.writeText(currentWallet.address)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000) // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy address:', err)
    }
  }

  const handleDonate = async () => {
    if (!amount || parseFloat(amount) <= 0) {
      setError('Please enter a valid amount')
      return
    }

    if (!wallets || wallets.length === 0 || !wallets[0].address) {
      setError('Wallet not connected. Please connect your wallet first.')
      return
    }

    if (!usdAmount) {
      setError('Failed to convert ETH to USD. Please try again.')
      return
    }

    if (!dbUser?.id) {
      setError('User not authenticated')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const wallet = wallets[0]
      const signer = await createSignerFromPrivyWallet(wallet)

      // Check if we're on Sepolia testnet
      const provider = signer.provider
      if (provider) {
        const network = await provider.getNetwork()
        // Sepolia testnet chain ID is 11155111
        if (network.chainId !== BigInt(11155111)) {
          // Try to switch automatically
          const wallet = wallets[0]
          const ethereumProvider = await wallet.getEthereumProvider()
          if (ethereumProvider) {
            try {
              await ethereumProvider.request({
                method: 'wallet_switchEthereumChain',
                params: [{ chainId: '0xaa36a7' }], // 11155111 in hex
              })
              // Wait for switch to complete
              await new Promise(resolve => setTimeout(resolve, 1000))
              // Re-check network
              const newNetwork = await provider.getNetwork()
              if (newNetwork.chainId !== BigInt(11155111)) {
                throw new Error('Please switch to Sepolia Testnet to make donations.')
              }
            } catch (switchError: any) {
              // If the chain doesn't exist, add it
              if (switchError.code === 4902 || switchError.code === -32603) {
                try {
                  await ethereumProvider.request({
                    method: 'wallet_addEthereumChain',
                    params: [
                      {
                        chainId: '0xaa36a7',
                        chainName: 'Sepolia',
                        nativeCurrency: {
                          name: 'Ether',
                          symbol: 'ETH',
                          decimals: 18,
                        },
                        rpcUrls: ['https://sepolia.infura.io/v3/b4880ead6a9a4f77a6de39dec6f3d0d0'],
                        blockExplorerUrls: ['https://sepolia.etherscan.io'],
                      },
                    ],
                  })
                  // Wait for add to complete
                  await new Promise(resolve => setTimeout(resolve, 1000))
                } catch (addError: any) {
                  throw new Error('Failed to add Sepolia network. Please add it manually in your wallet.')
                }
              } else if (switchError.code === 4001) {
                throw new Error('Network switch was rejected. Please switch to Sepolia Testnet manually.')
              } else {
                throw new Error('Please switch to Sepolia Testnet to make donations.')
              }
            }
          } else {
            throw new Error('Please switch to Sepolia Testnet to make donations.')
          }
        }
      }

      // Perform donation
      const hash = await donateToVault(vaultAddress, amount, signer)
      setTxHash(hash)

      // Save donation to database
      const response = await fetch('/api/donations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          disaster_event_id: eventId,
          amount: usdAmount,
          payment_reference: hash,
          user_id: dbUser.id,
        }),
      })

      if (!response.ok) {
        const result = await response.json()
        throw new Error(result.error || 'Failed to save donation record')
      }

      setStep('success')
      onDonationSuccess()
    } catch (err: any) {
      setError(err.message || 'Failed to process donation')
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    setStep('connect')
    setAmount('')
    setUsdAmount(null)
    setTxHash(null)
    setError(null)
    setLoading(false)
    userNavigatedBack.current = false
    onOpenChange(false)
  }

  // Prevent modal from closing on outside click or ESC - only close explicitly
  const handleOpenChange = (newOpen: boolean) => {
    // Only allow closing if explicitly called from handleClose
    // Ignore all other close attempts (outside click, ESC key, etc.)
    if (!newOpen) {
      // Don't close - the modal should only close via the Close button
      return
    }
    // Allow opening
    onOpenChange(newOpen)
  }

  const getEtherscanUrl = (hash: string) => {
    return `https://sepolia.etherscan.io/tx/${hash}`
  }

  const currentWallet = wallets && wallets.length > 0 ? wallets[0] : null

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent 
        className="sm:max-w-lg"
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
        showCloseButton={false}
      >
        <DialogHeader className="relative">
          {step === 'donate' && (
            <button
              onClick={() => {
                userNavigatedBack.current = true
                setStep('connect')
                setError(null)
              }}
              disabled={loading}
              className="absolute top-0 right-12 p-2.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-700 hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed border border-gray-200 hover:border-gray-300"
              title="Back to wallet connection"
            >
              <LogOut className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={handleClose}
            disabled={loading}
            className="absolute top-0 right-0 p-2.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-700 hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
          <DialogTitle className="text-2xl font-bold text-gray-900">
            {step === 'connect' && 'Connect Your Wallet'}
            {step === 'donate' && 'Make a Donation'}
            {step === 'success' && 'Donation Successful!'}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Step 1: Connect Wallet */}
          {step === 'connect' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-8">
                <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-4">
                  <Wallet className="w-8 h-8 text-green-800" />
                </div>
                <p className="text-center text-lg font-semibold text-gray-900 mb-2">
                  Connect your Ethereum wallet to make a donation
                </p>
                <p className="text-sm text-center text-gray-500">
                  We support MetaMask, WalletConnect, Coinbase Wallet, and more.
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}

              <Button
                onClick={handleConnectWallet}
                disabled={loading}
                className="w-full bg-green-800 hover:bg-green-900 text-lg py-6"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Wallet className="w-5 h-5 mr-2" />
                    Connect Wallet
                  </>
                )}
              </Button>
            </div>
          )}

          {/* Step 2: Enter Amount and Donate */}
          {step === 'donate' && (
            <div className="space-y-4">
              {currentWallet && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex flex-col items-center gap-3">
                    <p className="text-sm font-medium text-gray-700">Connected Wallet</p>
                    <div className="flex items-center gap-4 w-full">
                      {/* Wallet Address - Left Side */}
                      <div className="flex-1 flex items-center justify-center">
                        <button
                          onClick={handleCopyAddress}
                          className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-green-100 transition-colors group"
                        >
                          <span className="text-base font-mono text-green-800 font-semibold">
                            {truncateAddress(currentWallet.address)}
                          </span>
                          {copied ? (
                            <Check className="w-4 h-4 text-green-800" />
                          ) : (
                            <Copy className="w-4 h-4 text-gray-500 group-hover:text-green-800 transition-colors" />
                          )}
                        </button>
                      </div>
                      
                      {/* Dividing Line */}
                      <div className="h-8 w-px bg-green-300"></div>
                      
                      {/* ETH Balance - Right Side */}
                      <div className="flex-1 flex items-center justify-center">
                        <div className="flex flex-col items-center">
                          <div className="flex items-center gap-1.5 mb-1">
                            <img 
                              src="/ethereum.png" 
                              alt="Ethereum" 
                              className="w-3 h-3"
                            />
                            <p className="text-xs text-gray-600">Balance</p>
                          </div>
                          <p className="text-base font-semibold text-green-800">
                            {ethBalance !== null ? (
                              <>
                                {ethBalance} <span className="text-sm text-gray-600">ETH</span>
                              </>
                            ) : (
                              <span className="text-sm text-gray-500">Loading...</span>
                            )}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="amount" className="text-base font-medium">
                  Donation Amount
                </Label>
                <div className="relative">
                  <Input
                    id="amount"
                    type="number"
                    step="0.001"
                    min="0.0001"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    disabled={loading}
                    className="text-lg pr-20"
                  />
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
                    <img 
                      src="/ethereum.png" 
                      alt="Ethereum" 
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-500 font-black">
                      ETH
                    </span>
                  </div>
                </div>
                {usdAmount !== null && !converting && (
                  <div className="p-4 bg-gray-50 border-2 border-gray-300 rounded-lg">
                    <p className="text-xs text-gray-500 mb-1">Donation Amount</p>
                    <p className="text-3xl font-bold text-gray-900">
                      ${usdAmount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </p>
                    <p className="text-sm text-gray-600 mt-1">USD</p>
                    {ethPrice && (
                      <p className="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-200">
                        1 ETH = ${ethPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </p>
                    )}
                  </div>
                )}
                {converting && (
                  <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <p className="text-sm text-gray-600 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Converting to USD...
                    </p>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  Minimum: 0.0001 ETH
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}

              <Button
                onClick={handleDonate}
                disabled={loading || !amount || parseFloat(amount) <= 0 || !currentWallet}
                className="w-full bg-green-800 hover:bg-green-900 text-lg py-6"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Donate'
                )}
              </Button>
            </div>
          )}

          {/* Step 3: Success */}
          {step === 'success' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-8">
                <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-4">
                  <CheckCircle className="w-8 h-8 text-green-800" />
                </div>
                <p className="text-center text-lg font-semibold text-gray-900 mb-2">
                  Thank you for your donation!
                </p>
                {usdAmount && (
                  <p className="text-center text-2xl font-bold text-gray-900 mb-2">
                    ${usdAmount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </p>
                )}
                <p className="text-center text-sm text-gray-600">
                  Your contribution has been recorded successfully.
                </p>
              </div>

              {txHash && (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-xs text-gray-600 mb-2">Transaction Hash</p>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-mono text-gray-900 break-all flex-1">
                      {txHash}
                    </p>
                    <a
                      href={getEtherscanUrl(txHash)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-green-800 hover:text-green-900"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </div>
                </div>
              )}

              <Button
                onClick={handleClose}
                className="w-full bg-green-800 hover:bg-green-900 text-lg py-6"
              >
                Close
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
