import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/frontend/lib/supabase'

export const dynamic = 'force-dynamic'

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json()
    const { userId, userType, profile } = body

    if (!userId || !userType || !profile) {
      return NextResponse.json(
        { error: 'userId, userType, and profile are required' },
        { status: 400 }
      )
    }

    if (userType === 'user') {
      // Update user profile
      const { data, error } = await supabase
        .from('user_profiles')
        .update({
          name: profile.name,
          age: profile.age,
          location: profile.location,
          phone: profile.phone,
          bio: profile.bio,
          ...(profile.profile_pic_url !== undefined && { profile_pic_url: profile.profile_pic_url }),
        })
        .eq('id', userId)
        .select()
        .single()

      if (error) {
        console.error('Error updating user profile:', error)
        return NextResponse.json(
          { error: 'Failed to update profile', details: error.message },
          { status: 500 }
        )
      }

      return NextResponse.json({ success: true, data })
    } else if (userType === 'organization') {
      // Update organization profile
      const { data, error } = await supabase
        .from('organization_profiles')
        .update({
          name: profile.name,
          location: profile.location,
          registration_number: profile.registration_number,
          website_url: profile.website_url,
          phone: profile.phone,
          description: profile.description,
          established_date: profile.established_date,
          ...(profile.profile_pic_url !== undefined && { profile_pic_url: profile.profile_pic_url }),
        })
        .eq('id', userId)
        .select()
        .single()

      if (error) {
        console.error('Error updating organization profile:', error)
        return NextResponse.json(
          { error: 'Failed to update profile', details: error.message },
          { status: 500 }
        )
      }

      return NextResponse.json({ success: true, data })
    } else {
      return NextResponse.json(
        { error: 'Invalid userType. Must be "user" or "organization"' },
        { status: 400 }
      )
    }
  } catch (error: any) {
    console.error('Update profile error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    )
  }
}
