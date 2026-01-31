"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuthGuard } from "@/lib/auth-guard"
import { useAuth } from "@/lib/auth"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, User, Building2, CheckCircle2 } from "lucide-react"
import IndividualProfileForm from "@/components/individual-profile-form"
import OrganizationProfileForm from "@/components/organization-profile-form"
import Navbar from "@/components/navbar"

export default function SignupPage() {
  const router = useRouter()
  const { isAuthorized, isLoading: authLoading, profileComplete } = useAuthGuard({
    requireAuth: true,
    requireProfile: false, // Signup page doesn't require profile
  })
  const { user } = useAuth()
  const [userType, setUserType] = useState<'user' | 'organization' | null>(null)

  // Redirect if profile is already complete
  useEffect(() => {
    if (isAuthorized && !authLoading && profileComplete) {
      router.replace("/home")
    }
  }, [isAuthorized, authLoading, profileComplete, router])

  // Remove global top blur on mount
  useEffect(() => {
    document.body.classList.add('no-top-blur')
    return () => {
      document.body.classList.remove('no-top-blur')
    }
  }, [])

  if (authLoading || !isAuthorized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <Loader2 className="h-8 w-8 animate-spin text-green-800" />
      </div>
    )
  }

  // If user type is selected, show the appropriate form
  if (userType) {
    return (
      <div className="min-h-screen bg-gray-50 pt-28 pb-12 px-4">
        <Navbar />
        <div className="max-w-2xl mx-auto">
          <Button 
            variant="ghost" 
            onClick={() => setUserType(null)} 
            className="mb-6 hover:bg-transparent hover:text-green-800 text-gray-500 pl-0"
          >
            ← Back to selection
          </Button>
          
          <Card className="border-t-4 border-t-green-800 shadow-lg">
            <CardHeader className="text-center pb-2">
              <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4 text-green-800">
                {userType === 'user' ? <User className="h-8 w-8" /> : <Building2 className="h-8 w-8" />}
              </div>
              <CardTitle className="text-3xl text-gray-900">
                {userType === 'user' ? 'Individual Profile' : 'Organization Profile'}
            </CardTitle>
              <CardDescription className="text-lg text-gray-600 mt-2">
              {userType === 'user' 
                  ? 'Tell us a bit about yourself to join the community'
                  : 'Register your organization to start making an impact'}
            </CardDescription>
          </CardHeader>
            <CardContent className="p-8">
            {userType === 'user' ? (
              <IndividualProfileForm onComplete={() => router.push("/home")} />
            ) : (
              <OrganizationProfileForm onComplete={() => router.push("/home")} />
            )}
          </CardContent>
        </Card>
        </div>
      </div>
    )
  }

  // Show user type selection
  return (
    <div className="min-h-screen bg-gray-50 pt-28 pb-12 px-4 flex flex-col items-center">
      <Navbar />
      
      <div className="text-center mb-12 max-w-2xl">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 tracking-tight">
          Welcome to <span className="text-green-800">HealTheWorld</span>
        </h1>
        <p className="text-xl text-gray-600">
          Join our global community dedicated to making a difference. <br/>
          Choose how you would like to participate.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 w-full max-w-4xl">
        {/* Individual Option */}
        <div 
          className="group relative bg-white rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-green-200 cursor-pointer flex flex-col items-center text-center"
            onClick={() => setUserType('user')}
          >
          <div className="w-20 h-20 bg-green-50 rounded-full flex items-center justify-center mb-6 group-hover:bg-green-100 transition-colors">
            <User className="h-10 w-10 text-green-800" />
            </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-3 group-hover:text-green-800 transition-colors">
            I'm an Individual
          </h3>
          <p className="text-gray-500 mb-8 leading-relaxed">
            I want to discover causes, make donations, and track my impact on disaster relief efforts worldwide.
          </p>
          <div className="mt-auto">
            <Button className="w-full bg-white text-green-800 border-2 border-green-800 hover:bg-green-800 hover:text-white font-semibold py-6 rounded-xl transition-all">
              Continue as Individual
          </Button>
          </div>
        </div>

        {/* Organization Option */}
        <div 
          className="group relative bg-white rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-green-200 cursor-pointer flex flex-col items-center text-center"
            onClick={() => setUserType('organization')}
          >
          <div className="w-20 h-20 bg-green-50 rounded-full flex items-center justify-center mb-6 group-hover:bg-green-100 transition-colors">
            <Building2 className="h-10 w-10 text-green-800" />
            </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-3 group-hover:text-green-800 transition-colors">
            I'm an Organization
          </h3>
          <p className="text-gray-500 mb-8 leading-relaxed">
            I represent a registered organization that wants to create campaigns, receive funds, and provide aid.
          </p>
          <div className="mt-auto">
            <Button className="w-full bg-white text-green-800 border-2 border-green-800 hover:bg-green-800 hover:text-white font-semibold py-6 rounded-xl transition-all">
              Continue as Organization
          </Button>
          </div>
        </div>
      </div>

      <p className="mt-12 text-sm text-gray-400">
        By continuing, you agree to our Terms of Service and Privacy Policy.
      </p>
    </div>
  )
}
