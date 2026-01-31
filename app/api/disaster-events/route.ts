import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    const { data, error } = await supabase
      .from('disaster_events')
      .select('*')
      .order('occurred_at', { ascending: false })

    if (error) {
      console.error('Error fetching disaster events:', error)
      return NextResponse.json(
        { error: 'Failed to fetch disaster events', details: error.message },
        { status: 500 }
      )
    }

    // Ensure total_donations is properly parsed as a number for all events
    // PostgreSQL DECIMAL fields are returned as strings, so we need to parse them
    const eventsWithParsedTotals = (data || []).map(event => ({
      ...event,
      total_donations: event.total_donations !== null && event.total_donations !== undefined
        ? parseFloat(event.total_donations.toString())
        : 0
    }))

    return NextResponse.json({ success: true, data: eventsWithParsedTotals })
  } catch (error: any) {
    console.error('Fetch disaster events error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
