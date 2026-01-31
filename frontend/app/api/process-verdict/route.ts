import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { claimId, verdict, event, claim } = body

    if (!claimId || !verdict || !event || !claim) {
      return NextResponse.json(
        { error: 'Missing required fields: claimId, verdict, event, claim' },
        { status: 400 }
      )
    }

    // Determine verdict type
    const verdictType = verdict.toLowerCase()

    if (verdictType === 'accept') {
      // Fetch organization to get wallet address (if available in profile)
      const { data: orgProfile } = await supabase
        .from('organization_profiles')
        .select('*')
        .eq('id', claim.organization_id)
        .single()

      if (!orgProfile?.wallet_address) {
        return NextResponse.json(
          { error: 'Organization wallet address is required for accepting claims' },
          { status: 400 }
        )
      }

      // Call the external /vote endpoint
      const externalBaseUrl = process.env.NEXT_PUBLIC_BASE_URL
      if (!externalBaseUrl) {
        return NextResponse.json(
          { error: 'NEXT_PUBLIC_BASE_URL is not configured' },
          { status: 500 }
        )
      }

      const voteRequest = {
        wallet_address: orgProfile.wallet_address,
        amount: parseFloat(claim.requested_amount.toString()),
        vault_address: event.vault_address,
        vote: "release"
      }

      const voteResponse = await fetch(`${externalBaseUrl}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(voteRequest),
      })

      if (!voteResponse.ok) {
        const errorText = await voteResponse.text()
        console.error('Vote API error:', voteResponse.status, errorText)
        return NextResponse.json(
          { error: 'Failed to process vote', details: errorText },
          { status: voteResponse.status }
        )
      }

      const voteResult = await voteResponse.json()

      // Update claim with tx_hash and set status to accepted
      const { error: updateError } = await supabase
        .from('claims')
        .update({
          status: 'accepted',
          claim_tx_hash: voteResult.tx_hash,
          decided_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', claimId)

      if (updateError) {
        console.error('Error updating claim:', updateError)
        return NextResponse.json(
          { error: 'Failed to update claim', details: updateError.message },
          { status: 500 }
        )
      }

      return NextResponse.json({
        success: true,
        verdict: 'accept',
        voteResponse: voteResult,
      })
    } else if (verdictType === 'reject') {
      // Simply update status to rejected
      const { error: updateError } = await supabase
        .from('claims')
        .update({
          status: 'rejected',
          decided_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', claimId)

      if (updateError) {
        console.error('Error updating claim:', updateError)
        return NextResponse.json(
          { error: 'Failed to update claim', details: updateError.message },
          { status: 500 }
        )
      }

      return NextResponse.json({
        success: true,
        verdict: 'reject',
      })
    } else if (verdictType === 'raise_amount' || verdictType === 'lower_amount') {
      // Fetch event details for disaster_details
      const { data: eventDetails } = await supabase
        .from('disaster_events')
        .select('title, description, location, disaster_type')
        .eq('id', claim.disaster_event_id || event.id)
        .single()

      const disasterDetails = eventDetails 
        ? `${eventDetails.title}. ${eventDetails.description || ''} Location: ${eventDetails.location}. Type: ${eventDetails.disaster_type || 'Unknown'}.`
        : 'Disaster relief event details.'

      // Call the external /vote endpoint for higher/lower
      const externalBaseUrl = process.env.NEXT_PUBLIC_BASE_URL
      if (!externalBaseUrl) {
        return NextResponse.json(
          { error: 'NEXT_PUBLIC_BASE_URL is not configured' },
          { status: 500 }
        )
      }

      const voteRequest = {
        claim_submitted: claim.description,
        relief_fund: parseFloat(claim.requested_amount.toString()),
        disaster_details: disasterDetails,
        vote: verdictType === 'raise_amount' ? 'higher' : 'lower'
      }

      const voteResponse = await fetch(`${externalBaseUrl}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(voteRequest),
      })

      if (!voteResponse.ok) {
        const errorText = await voteResponse.text()
        console.error('Vote API error:', voteResponse.status, errorText)
        return NextResponse.json(
          { error: 'Failed to process vote', details: errorText },
          { status: voteResponse.status }
        )
      }

      const voteResult = await voteResponse.json()

      // Update requested_amount and reset votes
      const { error: updateError } = await supabase
        .from('claims')
        .update({
          requested_amount: voteResult.updated_relief_fund,
          status: 'pending',
          total_votes: 0,
          accept_votes: 0,
          reject_votes: 0,
          raise_amount_votes: 0,
          lower_amount_votes: 0,
          updated_at: new Date().toISOString(),
        })
        .eq('id', claimId)

      if (updateError) {
        console.error('Error updating claim:', updateError)
        return NextResponse.json(
          { error: 'Failed to update claim', details: updateError.message },
          { status: 500 }
        )
      }

      // Also need to delete all existing votes for this claim
      const { error: deleteVotesError } = await supabase
        .from('claim_votes')
        .delete()
        .eq('claim_id', claimId)

      if (deleteVotesError) {
        console.error('Error deleting votes:', deleteVotesError)
        // Don't fail the request, just log the error
      }

      return NextResponse.json({
        success: true,
        verdict: verdictType,
        voteResponse: voteResult,
      })
    } else {
      return NextResponse.json(
        { error: 'Invalid verdict type' },
        { status: 400 }
      )
    }
  } catch (error: any) {
    console.error('Process verdict error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
