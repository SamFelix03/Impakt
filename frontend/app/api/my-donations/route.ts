import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get('userId')

    if (!userId) {
      return NextResponse.json(
        { error: 'userId is required' },
        { status: 400 }
      )
    }

    // Fetch donations for the user
    const { data: donations, error } = await supabase
      .from('donations')
      .select('id, amount, donated_at, disaster_event_id')
      .eq('donor_id', userId)
      .order('donated_at', { ascending: false })

    if (error) {
      console.error('Error fetching donations:', error)
      return NextResponse.json(
        { error: 'Failed to fetch donations', details: error.message },
        { status: 500 }
      )
    }

    if (!donations || donations.length === 0) {
      return NextResponse.json({
        success: true,
        data: [],
      })
    }

    // Get unique disaster event IDs
    const eventIds = [...new Set(donations.map(d => d.disaster_event_id))]

    // Fetch disaster events
    const { data: events, error: eventsError } = await supabase
      .from('disaster_events')
      .select('*')
      .in('id', eventIds)

    if (eventsError) {
      console.error('Error fetching disaster events:', eventsError)
    }

    // Create a map of events by ID
    const eventsMap = new Map()
    if (events) {
      events.forEach(event => eventsMap.set(event.id, event))
    }

    // Parse amounts as numbers and join with event data
    const donationsWithParsedAmounts = donations.map(donation => {
      const event = eventsMap.get(donation.disaster_event_id)
      return {
        id: donation.id,
        amount: parseFloat(donation.amount.toString()),
        donated_at: donation.donated_at,
        disaster_event: event ? {
          id: event.id,
          title: event.title,
          location: event.location,
          disaster_type: event.disaster_type,
          severity: event.severity,
        } : null,
      }
    })

    return NextResponse.json(
      {
        success: true,
        data: donationsWithParsedAmounts,
      },
      {
        headers: {
          'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0',
        },
      }
    )
  } catch (error: any) {
    console.error('Fetch donations error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
