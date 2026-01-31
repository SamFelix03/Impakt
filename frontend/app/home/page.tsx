"use client"

import { useEffect, useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { useAuthGuard } from "@/frontend/lib/auth-guard"
import { useAuth } from "@/frontend/lib/auth"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/frontend/components/ui/card"
import { Button } from "@/frontend/components/ui/button"
import { Loader2, MapPin, Calendar, DollarSign, AlertCircle } from "lucide-react"
import Navbar from "@/frontend/components/navbar"
import { format } from "date-fns"

interface DisasterEvent {
  id: string
  title: string
  description: string | null
  location: string
  disaster_type: string | null
  severity: string | null
  occurred_at: string
  vault_address: string | null
  read_more_link: string | null
  target_amount: number | null
  tg_group_link: string | null
  total_donations: number
  created_at: string
  updated_at: string
}

function ProgressCard({ 
  event, 
  targetProgress, 
  onDonateClick,
  isOrganization = false
}: { 
  event: DisasterEvent
  targetProgress: number
  onDonateClick: () => void
  isOrganization?: boolean
}) {
  const [animatedProgress, setAnimatedProgress] = useState(0)
  const hasAnimated = useRef(false)

  useEffect(() => {
    if (!hasAnimated.current) {
      hasAnimated.current = true
      // Start animation from 0
      setAnimatedProgress(0)
      // Animate to target progress
      const duration = 1000 // 1 second
      const startTime = Date.now()
      const startProgress = 0

      const animate = () => {
        const elapsed = Date.now() - startTime
        const progress = Math.min(elapsed / duration, 1)
        const currentProgress = startProgress + (targetProgress - startProgress) * progress
        setAnimatedProgress(currentProgress)

        if (progress < 1) {
          requestAnimationFrame(animate)
        } else {
          setAnimatedProgress(targetProgress)
        }
      }

      requestAnimationFrame(animate)
    }
  }, [targetProgress])

  return (
    <Card className="hover:shadow-xl transition-all duration-300 border-t-4 border-t-green-800 flex flex-col h-full">
      <CardHeader className="text-center">
        <div className="flex flex-col items-center gap-3">
          {event.severity && (
            <span className={`px-3 py-1 rounded-full text-xs font-semibold tracking-wide uppercase ${
              event.severity === 'high' ? 'bg-red-100 text-red-800' :
              event.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
              'bg-green-100 text-green-800'
            }`}>
              {event.severity}
            </span>
          )}
          <CardTitle className="text-2xl text-gray-900 leading-tight">{event.title}</CardTitle>
        </div>
        {event.disaster_type && (
          <CardDescription className="text-sm text-gray-500 font-medium uppercase tracking-wider mt-2">
            {event.disaster_type}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex-1 flex flex-col items-center text-center space-y-6">
        <div className="flex items-center justify-center gap-2 text-green-800 bg-green-50 px-4 py-2 rounded-full">
          <MapPin className="w-4 h-4" />
          <span className="text-sm font-semibold">{event.location}</span>
        </div>

        {event.target_amount && (
          <div className="flex flex-col items-center gap-3">
            <div className="relative w-24 h-24">
              <svg className="transform -rotate-90 w-24 h-24">
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="6"
                  fill="none"
                  className="text-gray-100"
                />
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="6"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 40}`}
                  strokeDashoffset={`${2 * Math.PI * 40 * (1 - animatedProgress / 100)}`}
                  className="text-green-800 transition-all duration-1000 ease-out"
                  style={{ strokeLinecap: 'round' }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xl font-bold text-green-800">
                  {Math.round(animatedProgress)}%
                </span>
              </div>
            </div>
            <div className="space-y-1">
              <p className="text-2xl font-bold text-gray-900">
                ${event.total_donations.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">
                raised of <span className="font-bold">${event.target_amount.toLocaleString()}</span> goal
              </p>
            </div>
          </div>
        )}

        {event.description && (
          <p className="text-gray-600 text-sm leading-relaxed line-clamp-3 max-w-sm mx-auto">
            {event.description}
          </p>
        )}
      </CardContent>
      <CardFooter className="pt-2 pb-6 px-6">
        <Button 
          className="w-full bg-green-800 hover:bg-green-900 text-white font-semibold py-6 text-lg shadow-md hover:shadow-lg transition-all rounded-xl"
          onClick={onDonateClick}
        >
          {isOrganization ? "View" : "Donate now"}
        </Button>
      </CardFooter>
    </Card>
  )
}

export default function HomePage() {
  const router = useRouter()
  const { isAuthorized, isLoading: authLoading } = useAuthGuard({
    requireAuth: true,
    requireProfile: true,
  })
  const { orgProfile } = useAuth()
  const isOrganization = !!orgProfile
  const [disasterEvents, setDisasterEvents] = useState<DisasterEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isAuthorized) {
      fetchDisasterEvents()
    }
  }, [isAuthorized])

  const fetchDisasterEvents = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch('/api/disaster-events')
      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch disaster events')
      }

      setDisasterEvents(result.data || [])
    } catch (err: any) {
      console.error('Error fetching disaster events:', err)
      setError(err.message || 'Failed to load disaster events')
    } finally {
      setLoading(false)
    }
  }

  if (authLoading || !isAuthorized) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-green-800" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 pt-28 pb-8">
      <Navbar />
      
      <div className="container mx-auto px-4">
        <div className="mb-16 mt-8 text-center">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-4 tracking-tight">Make a Difference</h1>
          <p className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto">Help those in need by supporting disaster relief efforts</p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-green-800" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-16">
            <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
            <p className="text-red-600">{error}</p>
            <Button 
              onClick={fetchDisasterEvents} 
              className="mt-4 bg-green-800 hover:bg-green-900"
            >
              Try Again
            </Button>
          </div>
        ) : disasterEvents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16">
            <p className="text-gray-500 text-lg">No disaster events found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {disasterEvents.map((event) => {
              const targetProgress = event.target_amount 
                ? Math.min((event.total_donations / event.target_amount) * 100, 100)
                : 0
              
              return (
                <ProgressCard
                  key={event.id}
                  event={event}
                  targetProgress={targetProgress}
                  onDonateClick={() => router.push(`/event/${event.id}`)}
                  isOrganization={isOrganization}
                />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
