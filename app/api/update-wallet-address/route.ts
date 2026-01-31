import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, walletAddress } = body

    if (!userId) {
      return NextResponse.json(
        { error: 'userId is required' },
        { status: 400 }
      )
    }

    if (walletAddress === undefined) {
      return NextResponse.json(
        { error: 'walletAddress is required (can be null to disconnect)' },
        { status: 400 }
      )
    }

    // Update wallet_address in organization_profiles
    const { data, error } = await supabase
      .from('organization_profiles')
      .update({ wallet_address: walletAddress })
      .eq('id', userId)
      .select()
      .single()

    if (error) {
      console.error('Error updating wallet address:', error)
      return NextResponse.json(
        { error: 'Failed to update wallet address', details: error.message },
        { status: 500 }
      )
    }

    return NextResponse.json(
      {
        success: true,
        wallet_address: data.wallet_address,
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
    console.error('Update wallet address error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
