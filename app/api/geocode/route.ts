import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const address = searchParams.get('address')

    if (!address) {
      return NextResponse.json(
        { error: 'Address parameter is required' },
        { status: 400 }
      )
    }

    const apiKey = process.env.GOOGLE_MAPS_API_KEY
    if (!apiKey) {
      return NextResponse.json(
        { error: 'Google Maps API key not configured' },
        { status: 500 }
      )
    }

    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=${apiKey}`

    const response = await fetch(url)
    const data = await response.json()

    if (data.status === 'OK' && data.results && data.results.length > 0) {
      const location = data.results[0].geometry.location
      return NextResponse.json({
        success: true,
        coordinates: [location.lng, location.lat],
        formatted_address: data.results[0].formatted_address,
      })
    } else if (data.status === 'ZERO_RESULTS') {
      return NextResponse.json(
        { error: 'No results found for this address' },
        { status: 404 }
      )
    } else {
      return NextResponse.json(
        { error: `Geocoding failed: ${data.status}`, details: data.error_message },
        { status: 500 }
      )
    }
  } catch (error: any) {
    console.error('Geocoding error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
