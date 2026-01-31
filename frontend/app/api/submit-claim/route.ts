import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const content = formData.get('content') as string
    const vault_address = formData.get('vault_address') as string
    const disaster_event_id = formData.get('disaster_event_id') as string
    const organization_id = formData.get('organization_id') as string
    const images = formData.getAll('images') as File[]

    if (!content || !vault_address || !disaster_event_id || !organization_id) {
      return NextResponse.json(
        { error: 'Missing required fields: content, vault_address, disaster_event_id, organization_id' },
        { status: 400 }
      )
    }

    // Parse content to extract title and description
    // For now, we'll use a simple approach - first line as title, rest as description
    const lines = content.split('\n').filter(line => line.trim())
    const title = lines[0]?.trim() || 'Claim Request'
    const description = lines.slice(1).join('\n').trim() || content

    // Call the verification API
    const verifyFormData = new FormData()
    verifyFormData.append('content', content)
    verifyFormData.append('vault_address', vault_address)
    
    // Add images if provided
    images.forEach((image) => {
      if (image && image.size > 0) {
        verifyFormData.append('images', image)
      }
    })

    // Call the external verification service
    const externalBaseUrl = process.env.NEXT_PUBLIC_BASE_URL
    if (!externalBaseUrl) {
      return NextResponse.json(
        { error: 'NEXT_PUBLIC_BASE_URL is not configured' },
        { status: 500 }
      )
    }

    const verifyResponse = await fetch(`${externalBaseUrl}/verify`, {
      method: 'POST',
      body: verifyFormData,
    })

    if (!verifyResponse.ok) {
      const errorText = await verifyResponse.text()
      console.error('Verification API error:', verifyResponse.status, errorText)
      return NextResponse.json(
        { error: 'Failed to verify claim', details: errorText },
        { status: verifyResponse.status }
      )
    }

    const verifyResult = await verifyResponse.json()

    const recommendedAmount = verifyResult.recommended_amount_usd || 0

    // Ensure recommended amount is positive and valid
    const requestedAmount = Math.max(0.01, recommendedAmount)

    // Create the claim record atomically
    const { data: claim, error: claimError } = await supabase
      .from('claims')
      .insert({
        disaster_event_id,
        organization_id,
        title,
        description,
        requested_amount: requestedAmount,
        status: 'pending',
      })
      .select()
      .single()

    if (claimError) {
      console.error('Error creating claim:', claimError)
      return NextResponse.json(
        { error: 'Failed to create claim', details: claimError.message },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      claim,
      verification: verifyResult,
    })
  } catch (error: any) {
    console.error('Submit claim error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
