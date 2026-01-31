/**
 * Convert ETH to USD using our server-side API route
 * @param ethAmount Amount in ETH (as a number or string)
 * @returns USD value as a number
 */
export async function convertEthToUsd(ethAmount: number | string): Promise<number> {
  try {
    const ethValue = typeof ethAmount === 'string' ? parseFloat(ethAmount) : ethAmount
    
    if (isNaN(ethValue) || ethValue <= 0) {
      return 0
    }

    // Fetch current ETH price from our server-side API (avoids CORS and rate limiting)
    const ethPriceInUsd = await getEthPriceInUsd()
    return ethValue * ethPriceInUsd
  } catch (error: any) {
    console.error('Error converting ETH to USD:', error)
    throw new Error(`Failed to convert ETH to USD: ${error.message}`)
  }
}

/**
 * Get current ETH price in USD from our server-side API route
 * This avoids CORS issues and rate limiting
 * @returns ETH price in USD
 */
export async function getEthPriceInUsd(): Promise<number> {
  try {
    const response = await fetch('/api/eth-price', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store',
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      if (response.status === 429) {
        throw new Error('Rate limited. Please try again in a few moments.')
      }
      throw new Error(errorData.error || `API error: ${response.status}`)
    }

    const data = await response.json()
    const ethPriceInUsd = data.price

    if (!ethPriceInUsd || typeof ethPriceInUsd !== 'number') {
      throw new Error('Invalid response from ETH price API')
    }

    return ethPriceInUsd
  } catch (error: any) {
    console.error('Error fetching ETH price:', error)
    throw new Error(`Failed to fetch ETH price: ${error.message}`)
  }
}
