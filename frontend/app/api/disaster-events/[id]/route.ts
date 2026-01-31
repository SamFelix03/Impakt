import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: eventId } = await params

    // Fetch the disaster event
    const { data: event, error: eventError } = await supabase
      .from('disaster_events')
      .select('*')
      .eq('id', eventId)
      .single()

    if (eventError) {
      console.error('Error fetching disaster event:', eventError)
      return NextResponse.json(
        { error: 'Failed to fetch disaster event', details: eventError.message },
        { status: 500 }
      )
    }

    if (!event) {
      return NextResponse.json(
        { error: 'Event not found' },
        { status: 404 }
      )
    }

    // Parse total_donations as number (PostgreSQL DECIMAL fields are returned as strings)
    const eventWithParsedTotal = {
      ...event,
      total_donations: event.total_donations !== null && event.total_donations !== undefined
        ? parseFloat(event.total_donations.toString())
        : 0
    }

    // Fetch all donations for this event (for the "View all donors" modal)
    // According to schema.sql, donations table has: id, disaster_event_id, donor_id, amount, message, donated_at, payment_reference
    // Fetch all donations - Supabase default limit is 1000, but we'll fetch in batches if needed
    const { data: recentDonations, error: donationsError } = await supabase
      .from('donations')
      .select(`
        id,
        disaster_event_id,
        donor_id,
        amount,
        message,
        donated_at,
        payment_reference,
        donor:users!donations_donor_id_fkey(
          id,
          email,
          user_profiles(name, profile_pic_url),
          organization_profiles(name, profile_pic_url)
        )
      `)
      .eq('disaster_event_id', eventId)
      .order('donated_at', { ascending: false })
      // Remove limit to fetch all donations (Supabase will return up to 1000 by default)
      // If there are more than 1000, we'd need pagination, but that's unlikely for a single event

    if (donationsError) {
      console.error('Error fetching donations:', donationsError)
    }

    // Fetch top donors (top 4 by amount)
    const { data: topDonors, error: topDonorsError } = await supabase
      .from('donations')
      .select(`
        id,
        disaster_event_id,
        donor_id,
        amount,
        message,
        donated_at,
        payment_reference,
        donor:users!donations_donor_id_fkey(
          id,
          email,
          user_profiles(name, profile_pic_url),
          organization_profiles(name, profile_pic_url)
        )
      `)
      .eq('disaster_event_id', eventId)
      .order('amount', { ascending: false })
      .limit(4)

    if (topDonorsError) {
      console.error('Error fetching top donors:', topDonorsError)
    }

    // Fetch claims
    const { data: claims, error: claimsError } = await supabase
      .from('claims')
      .select(`
        *,
        organization:users!claims_organization_id_fkey(
          id,
          email,
          organization_profiles(name, profile_pic_url, registration_number)
        )
      `)
      .eq('disaster_event_id', eventId)
      .order('created_at', { ascending: false })

    if (claimsError) {
      console.error('Error fetching claims:', claimsError)
    }

    // Get total donation count
    const { count: donationCount, error: countError } = await supabase
      .from('donations')
      .select('*', { count: 'exact', head: true })
      .eq('disaster_event_id', eventId)

    if (countError) {
      console.error('Error fetching donation count:', countError)
    }

    return NextResponse.json(
      {
        success: true,
        event: eventWithParsedTotal,
        donations: recentDonations || [],
        topDonors: topDonors || [],
        claims: claims || [],
        donationCount: donationCount || 0,
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
    console.error('Fetch event details error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
