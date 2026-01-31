import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const organizationId = searchParams.get('organizationId')

    if (!organizationId) {
      return NextResponse.json(
        { error: 'organizationId is required' },
        { status: 400 }
      )
    }

    // Fetch claims for the organization
    const { data: claims, error } = await supabase
      .from('claims')
      .select('id, title, description, requested_amount, status, created_at, updated_at, disaster_event_id')
      .eq('organization_id', organizationId)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching claims:', error)
      return NextResponse.json(
        { error: 'Failed to fetch claims', details: error.message },
        { status: 500 }
      )
    }

    if (!claims || claims.length === 0) {
      return NextResponse.json({
        success: true,
        data: [],
      })
    }

    // Get unique disaster event IDs
    const eventIds = [...new Set(claims.map(c => c.disaster_event_id))]

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
    const claimsWithParsedAmounts = claims.map(claim => {
      const event = eventsMap.get(claim.disaster_event_id)
      return {
        id: claim.id,
        title: claim.title,
        description: claim.description,
        requested_amount: parseFloat(claim.requested_amount.toString()),
        status: claim.status,
        created_at: claim.created_at,
        updated_at: claim.updated_at,
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
        data: claimsWithParsedAmounts,
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
    console.error('Fetch claims error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
