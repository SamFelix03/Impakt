import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, userType } = body

    if (!userId || !userType) {
      return NextResponse.json(
        { error: 'userId and userType are required' },
        { status: 400 }
      )
    }

    if (userType !== 'user' && userType !== 'organization') {
      return NextResponse.json(
        { error: 'userType must be "user" or "organization"' },
        { status: 400 }
      )
    }

    const { data, error } = await supabase
      .from('users')
      .update({ user_type: userType })
      .eq('id', userId)
      .select()
      .single()

    if (error) {
      console.error('Error updating user type:', error)
      return NextResponse.json(
        { error: 'Failed to update user type', details: error.message },
        { status: 500 }
      )
    }

    return NextResponse.json({ success: true, data })
  } catch (error: any) {
    console.error('Update user type error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
