import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'

async function checkAndProcessVerdict(claimId: string, request: NextRequest) {
  try {
    // Fetch the claim with updated vote counts
    const { data: claim, error: claimError } = await supabase
      .from('claims')
      .select('*')
      .eq('id', claimId)
      .single()

    if (claimError || !claim) {
      console.error('Error fetching claim for verdict:', claimError)
      return
    }

    // Check if total votes reached 3
    if (claim.total_votes < 3) {
      return
    }

    // Determine verdict (highest vote count)
    const voteCounts = {
      accept: claim.accept_votes || 0,
      reject: claim.reject_votes || 0,
      raise_amount: claim.raise_amount_votes || 0,
      lower_amount: claim.lower_amount_votes || 0,
    }

    const maxVotes = Math.max(...Object.values(voteCounts))
    const verdict = Object.keys(voteCounts).find(key => voteCounts[key as keyof typeof voteCounts] === maxVotes)

    if (!verdict) {
      console.error('Could not determine verdict')
      return
    }

    // Fetch event details separately
    const { data: event, error: eventError } = await supabase
      .from('disaster_events')
      .select('*')
      .eq('id', claim.disaster_event_id)
      .single()

    if (eventError || !event) {
      console.error('Error fetching event for verdict:', eventError)
      return
    }

    // Call process-verdict endpoint (internal Next.js API route)
    // Use the request origin to construct the internal API URL
    const appUrl = request.nextUrl.origin
    const verdictResponse = await fetch(`${appUrl}/api/process-verdict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        claimId,
        verdict,
        event: {
          id: event.id,
          vault_address: event.vault_address,
          total_donations: event.total_donations,
        },
        claim: {
          id: claim.id,
          title: claim.title,
          description: claim.description,
          requested_amount: claim.requested_amount,
          organization_id: claim.organization_id,
        },
      }),
    })

    if (!verdictResponse.ok) {
      const error = await verdictResponse.json()
      console.error('Error processing verdict:', error)
    }
  } catch (error) {
    console.error('Error in checkAndProcessVerdict:', error)
  }
}

// GET: Fetch user votes for claims
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const claimIds = searchParams.get('claim_ids')?.split(',') || []
    const userId = searchParams.get('user_id')

    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      )
    }

    if (claimIds.length === 0) {
      return NextResponse.json({ success: true, votes: [] })
    }

    const { data: votes, error } = await supabase
      .from('claim_votes')
      .select('claim_id, vote')
      .eq('voter_id', userId)
      .in('claim_id', claimIds)

    if (error) {
      console.error('Error fetching user votes:', error)
      return NextResponse.json(
        { error: 'Failed to fetch votes', details: error.message },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      votes: votes || [],
    })
  } catch (error: any) {
    console.error('Fetch votes error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}

// POST: Create or update a vote
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { claim_id, user_id, vote } = body

    if (!user_id || !claim_id || !vote) {
      return NextResponse.json(
        { error: 'Missing required fields: user_id, claim_id, vote' },
        { status: 400 }
      )
    }

    if (!['accept', 'reject', 'raise_amount', 'lower_amount'].includes(vote)) {
      return NextResponse.json(
        { error: 'Invalid vote value. Must be: accept, reject, raise_amount, or lower_amount' },
        { status: 400 }
      )
    }

    // Check if user already voted on this claim
    const { data: existingVote, error: fetchError } = await supabase
      .from('claim_votes')
      .select('id, vote')
      .eq('claim_id', claim_id)
      .eq('voter_id', user_id)
      .single()

    if (fetchError && fetchError.code !== 'PGRST116') {
      console.error('Error checking existing vote:', fetchError)
      return NextResponse.json(
        { error: 'Failed to check existing vote', details: fetchError.message },
        { status: 500 }
      )
    }

    if (existingVote) {
      // Update existing vote
      const { data: updatedVote, error: updateError } = await supabase
        .from('claim_votes')
        .update({ vote, voted_at: new Date().toISOString() })
        .eq('id', existingVote.id)
        .select()
        .single()

      if (updateError) {
        console.error('Error updating vote:', updateError)
        return NextResponse.json(
          { error: 'Failed to update vote', details: updateError.message },
          { status: 500 }
        )
      }

      // Check if we need to process verdict (after vote counts are updated by trigger)
      await checkAndProcessVerdict(claim_id, request)

      return NextResponse.json({
        success: true,
        vote: updatedVote,
        action: 'updated',
      })
    } else {
      // Create new vote
      const { data: newVote, error: insertError } = await supabase
        .from('claim_votes')
        .insert({
          claim_id,
          voter_id: user_id,
          vote,
        })
        .select()
        .single()

      if (insertError) {
        console.error('Error creating vote:', insertError)
        return NextResponse.json(
          { error: 'Failed to create vote', details: insertError.message },
          { status: 500 }
        )
      }

      // Check if we need to process verdict (after vote counts are updated by trigger)
      await checkAndProcessVerdict(claim_id, request)

      return NextResponse.json({
        success: true,
        vote: newVote,
        action: 'created',
      })
    }
  } catch (error: any) {
    console.error('Create/update vote error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
