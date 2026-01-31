'use client'

import { usePrivy } from '@privy-io/react-auth'
import { useEffect, useState } from 'react'
import { supabase, type User, type UserProfile, type OrganizationProfile } from './supabase'

export function useAuth() {
  const { ready, authenticated, user, login, logout } = usePrivy()
  const [dbUser, setDbUser] = useState<User | null>(null)
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null)
  const [orgProfile, setOrgProfile] = useState<OrganizationProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [profileComplete, setProfileComplete] = useState(false)

  useEffect(() => {
    if (ready && authenticated && user) {
      syncUser()
    } else {
      setDbUser(null)
      setUserProfile(null)
      setOrgProfile(null)
      setProfileComplete(false)
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, authenticated, user])

  const syncUser = async () => {
      if (!user?.id) {
        return
      }

    setLoading(true)
    try {
      // Get user's email from Privy
      const email = user.email?.address || user.google?.email || ''
      const privyId = user.id // Privy DID

      // Check if user exists in database by privy_id
      const { data: existingUser, error: fetchError } = await supabase
        .from('users')
        .select('*')
        .eq('privy_id', privyId)
        .single()

      if (fetchError && fetchError.code !== 'PGRST116') {
        // Error other than "not found"
        console.error('[Auth] Error fetching user:', fetchError)
        setLoading(false)
        return
      }


      if (existingUser) {
        // User exists, check for profile
        setDbUser(existingUser)
        
        // Check for user profile (individual)
        if (existingUser.user_type === 'user') {
          const { data: profile } = await supabase
            .from('user_profiles')
            .select('*')
            .eq('id', existingUser.id)
            .single()
          
          if (profile) {
            setUserProfile(profile)
            setProfileComplete(true)
          } else {
            setUserProfile(null)
            setProfileComplete(false)
          }
        }
        
        // Check for organization profile
        if (existingUser.user_type === 'organization') {
          const { data: profile } = await supabase
            .from('organization_profiles')
            .select('*')
            .eq('id', existingUser.id)
            .single()
          
          if (profile) {
            setOrgProfile(profile)
            setProfileComplete(true)
          } else {
            setOrgProfile(null)
            setProfileComplete(false)
          }
        }
      } else {
        // User doesn't exist, create new user record
        // Database will auto-generate UUID for id, we store privy_id separately
        const { data: newUser, error: createError } = await supabase
          .from('users')
          .insert({
            privy_id: privyId,
            email: email,
            user_type: 'user', // Default, will be updated during onboarding
          })
          .select()
          .single()

        if (createError) {
          console.error('[Auth] Error creating user:', createError)
          setLoading(false)
          return
        } else {
          setDbUser(newUser)
          setProfileComplete(false) // New user needs to complete profile
        }
      }
    } catch (error) {
      console.error('Error syncing user:', error)
    } finally {
      setLoading(false)
    }
  }

  return {
    ready,
    authenticated,
    user,
    dbUser,
    userProfile,
    orgProfile,
    profileComplete,
    loading,
    login,
    logout,
    syncUser,
  }
}
