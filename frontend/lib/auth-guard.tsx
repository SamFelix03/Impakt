'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from './auth'

type AuthGuardOptions = {
  /** If true, redirects unauthenticated users to home */
  requireAuth?: boolean
  /** If true, redirects users without complete profiles to signup */
  requireProfile?: boolean
  /** Custom redirect path for unauthenticated users (default: '/') */
  redirectTo?: string
  /** Custom redirect path for incomplete profiles (default: '/signup') */
  signupRedirect?: string
}

/**
 * Centralized auth guard hook that handles all authentication and redirect logic
 * Use this in protected pages to ensure proper auth flow
 */
export function useAuthGuard(options: AuthGuardOptions = {}) {
  const {
    requireAuth = true,
    requireProfile = true,
    redirectTo = '/',
    signupRedirect = '/signup',
  } = options

  const router = useRouter()
  const pathname = usePathname()
  const { ready, authenticated, profileComplete, loading } = useAuth()
  
  const [isAuthorized, setIsAuthorized] = useState(false)
  const hasRedirected = useRef(false)
  const authCheckComplete = useRef(false)
  const redirectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const profileCompleteRef = useRef(profileComplete)

  // Keep ref in sync with latest profileComplete value
  // If profileComplete becomes true, cancel any pending redirects and authorize
  useEffect(() => {
    profileCompleteRef.current = profileComplete
    
    // If profile just completed and we have a pending redirect, cancel it
    if (profileComplete && redirectTimeoutRef.current) {
      clearTimeout(redirectTimeoutRef.current)
      redirectTimeoutRef.current = null
      hasRedirected.current = false
      
      // If user is authenticated and profile is complete, authorize
      if (ready && !loading && authenticated) {
        setIsAuthorized(true)
      }
    }
  }, [profileComplete, ready, loading, authenticated])

  useEffect(() => {
    // Clear any pending redirects
    if (redirectTimeoutRef.current) {
      clearTimeout(redirectTimeoutRef.current)
      redirectTimeoutRef.current = null
    }

    // Wait for auth to be ready AND not loading
    if (!ready || loading) {
      setIsAuthorized(false)
      authCheckComplete.current = false
      return
    }

    // Mark that we've completed an auth check
    authCheckComplete.current = true

    // If auth is not required, allow access immediately
    if (!requireAuth) {
      setIsAuthorized(true)
      hasRedirected.current = false
      return
    }

    // Check authentication
    if (!authenticated) {
      // Only redirect if we're not already on the redirect page
      if (!hasRedirected.current && pathname !== redirectTo) {
        hasRedirected.current = true
        router.replace(redirectTo)
      }
      setIsAuthorized(false)
      return
    }

    // Check profile completion if required
    // IMPORTANT: Wait a bit after loading completes to ensure profile check has finished
    // React state updates are batched, so profileComplete might not be updated immediately
    if (requireProfile && !profileComplete) {
      // Only redirect if we're not already on the signup page
      if (!hasRedirected.current && pathname !== signupRedirect) {
        // Use a delay to ensure profile check has fully completed
        // This prevents redirecting when profileComplete is just temporarily false
        redirectTimeoutRef.current = setTimeout(() => {
          // Check the latest value from ref (not stale closure)
          if (!profileCompleteRef.current) {
            hasRedirected.current = true
            router.replace(signupRedirect)
          } else {
            // Profile completed during delay - authorize instead
            setIsAuthorized(true)
            hasRedirected.current = false
          }
        }, 250) // Delay to ensure profile check completes and state updates
      }
      setIsAuthorized(false)
      return
    }

    // All checks passed - user is authorized
    setIsAuthorized(true)
    hasRedirected.current = false // Reset redirect flag when authorized
  }, [ready, loading, authenticated, profileComplete, requireAuth, requireProfile, redirectTo, signupRedirect, pathname, router])

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (redirectTimeoutRef.current) {
        clearTimeout(redirectTimeoutRef.current)
      }
    }
  }, [])

  return {
    isAuthorized,
    isLoading: !ready || loading || !authCheckComplete.current,
    ready,
    authenticated,
    profileComplete,
    loading,
  }
}
