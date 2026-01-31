import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

// Cache for ETH price to avoid excessive API calls
// Using a module-level variable (will be shared across requests in the same process)
let priceCache: { price: number; timestamp: number } | null = null
const CACHE_TTL = 1000 * 60 * 5 // 5 minutes

export async function GET(request: NextRequest) {
  try {
    // Check cache first
    if (priceCache && (Date.now() - priceCache.timestamp < CACHE_TTL)) {
      return NextResponse.json({ 
        success: true, 
        price: priceCache.price,
        cached: true 
      })
    }

    const response = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd',
      {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        // Add a small delay to avoid rate limiting
        next: { revalidate: 300 } // Cache for 5 minutes
      }
    )

    if (!response.ok) {
      // If rate limited, return a fallback price or error
      if (response.status === 429) {
        return NextResponse.json(
          { 
            error: 'Rate limited. Please try again later.',
            fallback: true 
          },
          { status: 429 }
        )
      }
      throw new Error(`CoinGecko API error: ${response.status}`)
    }

    const data = await response.json()
    const ethPriceInUsd = data.ethereum?.usd

    if (!ethPriceInUsd || typeof ethPriceInUsd !== 'number') {
      throw new Error('Invalid response from CoinGecko API')
    }

    // Update cache
    priceCache = { price: ethPriceInUsd, timestamp: Date.now() }

    return NextResponse.json({ 
      success: true, 
      price: ethPriceInUsd,
      cached: false 
    })
  } catch (error: any) {
    console.error('Error fetching ETH price:', error)
    return NextResponse.json(
      { 
        error: 'Failed to fetch ETH price', 
        details: error.message 
      },
      { status: 500 }
    )
  }
}
