"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/frontend/lib/auth"
import { Loader2 } from "lucide-react"

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { ready, authenticated, profileComplete, loading } = useAuth()

  useEffect(() => {
    if (ready && !authenticated) {
      // User not authenticated, stay on current page (they can see public content)
      return
    }

    if (ready && authenticated && !profileComplete) {
      // User authenticated but profile not complete, redirect to signup
      router.push("/signup")
    }
  }, [ready, authenticated, profileComplete, router])

  if (!ready || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  // If authenticated but profile not complete, show loading while redirecting
  if (authenticated && !profileComplete) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return <>{children}</>
}
