import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { disaster_event_id, amount, payment_reference, message, user_id } = body

    if (!user_id) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      )
    }

    if (!disaster_event_id || !amount || !payment_reference) {
      return NextResponse.json(
        { error: 'Missing required fields: disaster_event_id, amount, payment_reference' },
        { status: 400 }
      )
    }

    const donationAmount = parseFloat(amount)

    // Create donation record (amount is already in USD from the client)
    // According to schema.sql, the trigger_update_disaster_donations trigger
    // should automatically update total_donations when a donation is inserted
    const { data: donation, error: donationError } = await supabase
      .from('donations')
      .insert({
        disaster_event_id,
        donor_id: user_id,
        amount: donationAmount,
        payment_reference,
        message: message || null,
      })
      .select()
      .single()

    if (donationError) {
      console.error('Error creating donation:', donationError)
      return NextResponse.json(
        { error: 'Failed to create donation record', details: donationError.message },
        { status: 500 }
      )
    }

    // The trigger_update_disaster_donations trigger automatically updates total_donations
    // No manual update needed

    return NextResponse.json({
      success: true,
      donation,
    })
  } catch (error: any) {
    console.error('Create donation error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
