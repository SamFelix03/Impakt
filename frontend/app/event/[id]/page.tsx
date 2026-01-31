"use client"

import { useEffect, useState, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import { useAuthGuard } from "@/frontend/lib/auth-guard"
import { Card, CardContent, CardDescription, CardHeader } from "@/frontend/components/ui/card"
import { Button } from "@/frontend/components/ui/button"
import { Loader2, MapPin, ArrowLeft, Share2, CheckCircle, XCircle, TrendingUp, TrendingDown } from "lucide-react"
import Navbar from "@/frontend/components/navbar"
import { format } from "date-fns"
import { Map, MapMarker, MarkerContent, MapControls } from "@/frontend/components/ui/map"
import { geocodeLocation } from "@/frontend/lib/geocode"

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/frontend/components/ui/dialog"
import { Badge } from "@/frontend/components/ui/badge"
import { DonationModal } from "@/frontend/components/donation-modal"
import { SubmitClaimModal } from "@/frontend/components/submit-claim-modal"
import { useAuth } from "@/frontend/lib/auth"

interface Claim {
  id: string
  title: string
  description: string
  requested_amount: number
  approved_amount: number | null
  status: 'pending' | 'accepted' | 'rejected' | 'raise_amount' | 'lower_amount'
  total_votes: number
  accept_votes: number
  reject_votes: number
  raise_amount_votes: number
  lower_amount_votes: number
  claim_tx_hash: string | null
  organization: {
    id: string
    email: string
    organization_profiles: {
      name: string
      profile_pic_url: string | null
      registration_number: string | null
    } | null
  }
}

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

interface Donation {
  id: string
  amount: number
  message: string | null
  donated_at: string
  donor: {
    id: string
    email: string
    user_profiles?: { name: string; profile_pic_url: string | null } | null
    organization_profiles?: { name: string; profile_pic_url: string | null } | null
  } | null
}

export default function EventDetailsPage() {
  const router = useRouter()
  const params = useParams()
  const { isAuthorized, isLoading: authLoading } = useAuthGuard({
    requireAuth: true,
    requireProfile: true,
  })
  const { dbUser, orgProfile } = useAuth()
  const isOrganization = dbUser?.user_type === 'organization'
  const [event, setEvent] = useState<DisasterEvent | null>(null)
  const [donations, setDonations] = useState<Donation[]>([])
  const [topDonors, setTopDonors] = useState<Donation[]>([])
  const [claims, setClaims] = useState<Claim[]>([])
  
  // Check if organization has already submitted a claim for this event
  const hasSubmittedClaim = isOrganization && dbUser?.id 
    ? claims.some(claim => claim.organization.id === dbUser.id)
    : false
  const [donationCount, setDonationCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [mapCoords, setMapCoords] = useState<[number, number] | null>(null)
  const [isDonorsModalOpen, setIsDonorsModalOpen] = useState(false)
  const [isDonationModalOpen, setIsDonationModalOpen] = useState(false)
  const [isVoteModalOpen, setIsVoteModalOpen] = useState(false)
  const [isSubmitClaimModalOpen, setIsSubmitClaimModalOpen] = useState(false)
  const [selectedClaimForVote, setSelectedClaimForVote] = useState<Claim | null>(null)
  const [userVotes, setUserVotes] = useState<Record<string, 'accept' | 'reject' | 'raise_amount' | 'lower_amount'>>({})
  const hasAttemptedFetch = useRef(false)

  // Fetch event details once we have the event ID and user is authorized
  useEffect(() => {
    if (params.id && isAuthorized && !hasAttemptedFetch.current) {
      hasAttemptedFetch.current = true
      fetchEventDetails()
    }
  }, [params.id, isAuthorized])

  const fetchEventDetails = async (options?: { silent?: boolean }) => {
    try {
      // Avoid unmounting the whole page (and the donation modal) during background refreshes
      if (!options?.silent) {
        setLoading(true)
      }
      setError(null)
      
      // Add cache-busting timestamp to prevent caching
      const response = await fetch(`/api/disaster-events/${params.id}?t=${Date.now()}`, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
        },
      })
      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch event details')
      }

      // Ensure total_donations is a number (parse if needed, same as home page)
      const eventData = result.event
      if (eventData && eventData.total_donations !== null && eventData.total_donations !== undefined) {
        eventData.total_donations = typeof eventData.total_donations === 'string' 
          ? parseFloat(eventData.total_donations) 
          : Number(eventData.total_donations)
      }
      setEvent(eventData)
      // Ensure donations array is properly set - log for debugging
      const donationsArray = result.donations || []
      console.log(`[Event Details] Fetched ${donationsArray.length} donations for event ${params.id}`)
      setDonations(donationsArray)
      setTopDonors(result.topDonors || [])
      setClaims(result.claims || [])
      setDonationCount(result.donationCount || 0)
      
      // Fetch user votes for claims
      if (dbUser?.id) {
        fetchUserVotes(result.claims || [])
      }
      
      // Geocode location for map
      if (result.event?.location) {
        geocodeLocation(result.event.location).then((coords) => {
          setMapCoords(coords)
        })
      }
    } catch (err: any) {
      console.error('Error fetching event details:', err)
      setError(err.message || 'Failed to load event details')
    } finally {
      if (!options?.silent) {
        setLoading(false)
      }
    }
  }

  const fetchUserVotes = async (claimsList: Claim[]) => {
    if (!dbUser?.id || claimsList.length === 0) return
    
    try {
      const claimIds = claimsList.map(c => c.id)
      const response = await fetch(`/api/claim-votes?claim_ids=${claimIds.join(',')}&user_id=${dbUser.id}`)
      const result = await response.json()
      
      if (response.ok && result.votes) {
        const votesMap: Record<string, 'accept' | 'reject' | 'raise_amount' | 'lower_amount'> = {}
        result.votes.forEach((vote: any) => {
          votesMap[vote.claim_id] = vote.vote
        })
        setUserVotes(votesMap)
      }
    } catch (err) {
      console.error('Error fetching user votes:', err)
    }
  }

  const handleVote = async (claimId: string, vote: 'accept' | 'reject' | 'raise_amount' | 'lower_amount') => {
    if (!dbUser?.id) return

    try {
      const response = await fetch('/api/claim-votes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          claim_id: claimId,
          user_id: dbUser.id,
          vote: vote,
        }),
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to submit vote')
      }

      // Update user votes state
      setUserVotes(prev => ({
        ...prev,
        [claimId]: vote,
      }))

      // Refresh claims to get updated vote counts
      fetchEventDetails({ silent: true })
    } catch (err: any) {
      console.error('Error submitting vote:', err)
      alert(err.message || 'Failed to submit vote')
    }
  }

  const getDonorName = (donation: Donation) => {
    if (donation.donor?.user_profiles?.name) {
      return donation.donor.user_profiles.name
    }
    if (donation.donor?.organization_profiles?.name) {
      return donation.donor.organization_profiles.name
    }
    return donation.donor?.email?.split('@')[0] || 'Anonymous'
  }

  const getDonorInitial = (donation: Donation) => {
    const name = getDonorName(donation)
    return name.charAt(0).toUpperCase()
  }

  const getTopDonation = () => {
    if (donations.length === 0) return null
    return donations.reduce((top, current) => 
      current.amount > (top?.amount || 0) ? current : top
    )
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount)
  }

  const getStatusBadge = (status: Claim['status']) => {
    switch (status) {
      case 'accepted':
        return <Badge className="bg-green-100 text-green-800 hover:bg-green-200 border-green-200 text-base font-bold px-4 py-2">Accepted</Badge>
      case 'rejected':
        return <Badge className="bg-red-100 text-red-800 hover:bg-red-200 border-red-200 text-base font-bold px-4 py-2">Rejected</Badge>
      case 'pending':
        return <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-200 border-yellow-200 text-base font-bold px-4 py-2">Pending</Badge>
      case 'raise_amount':
        return <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-200 border-blue-200 text-base font-bold px-4 py-2">Raise Amount</Badge>
      case 'lower_amount':
        return <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-200 border-orange-200 text-base font-bold px-4 py-2">Lower Amount</Badge>
      default:
        return <Badge variant="outline" className="text-base font-bold px-4 py-2">Unknown</Badge>
    }
  }

  const progressPercentage = event && event.target_amount 
    ? Math.min((event.total_donations / event.target_amount) * 100, 100)
    : 0

  // Show loading while auth is checking or user is not authorized (will redirect)
  if (authLoading || !isAuthorized) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-green-800" />
      </div>
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 pt-28 pb-8">
        <Navbar />
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-8 w-8 animate-spin text-green-800" />
        </div>

        {/* Keep the donation modal mounted even if the page is refreshing */}
        {event?.vault_address && (
          <DonationModal
            open={isDonationModalOpen}
            onOpenChange={setIsDonationModalOpen}
            vaultAddress={event.vault_address}
            eventId={event.id}
            onDonationSuccess={() => {
              fetchEventDetails({ silent: true })
            }}
          />
        )}
      </div>
    )
  }

  if (error || !event) {
    return (
      <div className="min-h-screen bg-gray-50 pt-28 pb-8">
        <Navbar />
        <div className="container mx-auto px-4">
          <Button
            variant="ghost"
            onClick={() => router.push("/home")}
            className="mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Events
          </Button>
          <div className="flex flex-col items-center justify-center py-16">
            <p className="text-red-600 mb-4">{error || 'Event not found'}</p>
            <Button onClick={() => router.push("/home")} className="bg-green-800 hover:bg-green-900">
              Go Back
            </Button>
          </div>
        </div>
      </div>
    )
  }

  const topDonation = getTopDonation()

  return (
    <div className="min-h-screen bg-gray-50 pt-28 pb-8">
      <Navbar />
      
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Title */}
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
              {event.title}
            </h1>

            {/* Location */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-green-800">
                <MapPin className="w-5 h-5" />
                <span className="text-lg font-medium">{event.location}</span>
              </div>
              <button
                onClick={() => {
                  if (navigator.share) {
                    navigator.share({
                      title: event.title,
                      text: event.description || '',
                      url: window.location.href,
                    })
                  } else {
                    navigator.clipboard.writeText(window.location.href)
                  }
                }}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-700 hover:text-gray-900"
                title="Share"
              >
                <Share2 className="w-5 h-5" />
              </button>
            </div>

            {/* Map */}
            {mapCoords && (
              <Card className="overflow-hidden py-0">
                <CardContent className="p-0">
                  <div className="w-full h-96 relative">
                    <Map 
                      center={mapCoords} 
                      zoom={mapCoords[0] === 0 && mapCoords[1] === 20 ? 2 : 6}
                    >
                      <MapMarker longitude={mapCoords[0]} latitude={mapCoords[1]}>
                        <MarkerContent>
                          <div className="relative">
                            {/* Glowing red marker */}
                            <div className="absolute inset-0 animate-pulse">
                              <div className="w-8 h-8 bg-red-500 rounded-full opacity-75 blur-md"></div>
                            </div>
                            <div className="relative w-6 h-6 bg-red-600 rounded-full border-2 border-white shadow-lg">
                              <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-75"></div>
                            </div>
                          </div>
                        </MarkerContent>
                      </MapMarker>
                      <MapControls position="bottom-right" showZoom={true} />
                    </Map>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Description */}
            <Card>
              <CardContent className="p-6">
                <div className="prose max-w-none">
                  <p className="text-gray-700 whitespace-pre-wrap leading-relaxed text-base font-medium">
                    {event.description || 'No description available.'}
                  </p>
                  {event.read_more_link && (
                    <div className="mt-4">
                      <a
                        href={event.read_more_link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-green-800 hover:text-green-900 font-medium underline inline-flex items-center gap-1"
                      >
                        Read more
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Claims Section */}
            <div className="w-full max-w-full overflow-hidden">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Organization Fund Claims</h2>
              {claims.length > 0 ? (
                <div className="grid gap-4 w-full max-w-full">
                  {claims.map((claim) => (
                    <Card key={claim.id} className="w-full max-w-full overflow-hidden">
                      <CardHeader className="pb-4 w-full max-w-full">
                        <div className="flex justify-between items-start gap-4 w-full max-w-full mb-3">
                          <div className="flex-1 min-w-0">
                            <h3 className="text-xl font-bold text-gray-900 break-words mb-2">{claim.title}</h3>
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">by</span>
                              {claim.organization.organization_profiles?.profile_pic_url && (
                                <img 
                                  src={claim.organization.organization_profiles.profile_pic_url} 
                                  alt={claim.organization.organization_profiles.name}
                                  className="w-6 h-6 rounded-full object-cover flex-shrink-0"
                                />
                              )}
                              <span className="text-sm font-semibold text-gray-800 break-words">
                                {claim.organization.organization_profiles?.name || 'Unknown Organization'}
                              </span>
                            </div>
                          </div>
                          <div className="flex-shrink-0">
                          {getStatusBadge(claim.status)}
                        </div>
                          </div>
                        <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                            <div>
                            <span className="text-xs text-gray-500 uppercase tracking-wide font-semibold block mb-1">Allocated Relief Amount</span>
                            <span className="text-2xl font-bold text-green-800">{formatCurrency(claim.requested_amount)}</span>
                            </div>
                          {!isOrganization && (
                            <div className="flex flex-col items-end gap-3">
                              {claim.status === 'pending' && (
                                <>
                                  <div className="flex gap-3 text-xs">
                                    <div className="whitespace-nowrap">
                                      <span className="text-green-600 font-semibold">{claim.accept_votes}</span>
                                      <span className="text-gray-500 ml-1">Accept</span>
                            </div>
                                    <div className="whitespace-nowrap">
                                      <span className="text-red-600 font-semibold">{claim.reject_votes}</span>
                                      <span className="text-gray-500 ml-1">Reject</span>
                          </div>
                                  </div>
                                  <Button
                                    size="lg"
                                    className="bg-green-800 hover:bg-green-900 text-white font-bold text-base px-8 py-6 shadow-lg hover:shadow-xl transition-all"
                                    onClick={() => {
                                      setSelectedClaimForVote(claim)
                                      setIsVoteModalOpen(true)
                                    }}
                                  >
                                    Vote Now
                                  </Button>
                                </>
                              )}
                              {claim.status === 'accepted' && (
                                <div className="flex flex-col items-end gap-2">
                                  <div className="text-lg font-bold text-green-800">Claim Fulfilled!</div>
                                  {claim.claim_tx_hash && (
                                    <Button
                                      size="lg"
                                      variant="outline"
                                      className="border-green-800 text-green-800 hover:bg-green-50 font-semibold text-sm px-6 py-3"
                                      onClick={() => {
                                        window.open(`https://sepolia.etherscan.io/tx/0x${claim.claim_tx_hash}`, '_blank')
                                      }}
                                    >
                                      View Claim TX
                                    </Button>
                                  )}
                                </div>
                              )}
                              {claim.status === 'rejected' && (
                                <div className="text-lg font-bold text-red-800">Claim Rejected</div>
                              )}
                            </div>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="w-full max-w-full overflow-hidden pt-0">
                        <div className="mb-4">
                          <span className="text-xs text-gray-500 uppercase tracking-wide font-semibold block mb-2">Description</span>
                          <p className="text-base text-gray-700 leading-relaxed break-words whitespace-pre-wrap overflow-wrap-anywhere word-break-break-all w-full max-w-full">{claim.description}</p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Card className="bg-gray-50 border-dashed w-full max-w-full">
                  <CardContent className="flex flex-col items-center justify-center py-8 text-center">
                    <p className="text-gray-500">No claims found for this event yet.</p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* Right Column - Donation Summary */}
          <div className="space-y-6">
            <Card className="sticky top-28">
              <CardHeader>
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-800 mb-2">
                    {donationCount} donations
                  </p>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Progress Circle */}
                <div className="flex justify-center">
                  <div className="relative w-32 h-32">
                    <svg className="transform -rotate-90 w-32 h-32">
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        className="text-gray-200"
                      />
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${2 * Math.PI * 56}`}
                        strokeDashoffset={`${2 * Math.PI * 56 * (1 - progressPercentage / 100)}`}
                        className="text-green-800 transition-all duration-300"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-2xl font-bold text-green-800">
                        {Math.round(progressPercentage)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Amount Raised */}
                <div className="text-center">
                  <p className="text-3xl font-bold text-gray-900">
                    ${event.total_donations.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </p>
                  <p className="text-sm text-gray-600">
                    raised of {event.target_amount ? `$${event.target_amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'goal'}
                  </p>
                </div>

                {/* Action Buttons */}
                <div className="space-y-3">
                  <Button
                    className="w-full bg-green-800 hover:bg-green-900 text-lg py-6"
                    onClick={() => {
                      if (!event?.vault_address) {
                        alert('Vault address not available for this event')
                        return
                      }
                      setIsDonationModalOpen(true)
                    }}
                    disabled={!event?.vault_address}
                  >
                    Donate now
                  </Button>
                  {event.tg_group_link && (
                    <Button
                      variant="outline"
                      className="w-full border-green-800 text-green-800 hover:bg-green-50 text-lg py-6"
                      onClick={() => {
                        window.open(event.tg_group_link!, '_blank')
                      }}
                    >
                      Join Community
                    </Button>
                  )}
                  {isOrganization && (
                  <Button
                    variant="outline"
                      className="w-full border-green-800 text-green-800 hover:bg-green-50 text-lg py-6 disabled:opacity-50 disabled:cursor-not-allowed"
                      onClick={() => {
                        if (!orgProfile?.wallet_address) {
                          router.push('/dashboard')
                          return
                        }
                        setIsSubmitClaimModalOpen(true)
                      }}
                      disabled={!event?.vault_address || hasSubmittedClaim}
                      title={
                        hasSubmittedClaim 
                          ? "You have already submitted a claim for this event" 
                          : !orgProfile?.wallet_address
                          ? "Please connect your wallet in the dashboard to submit a claim"
                          : undefined
                      }
                    >
                      {hasSubmittedClaim 
                        ? "Claim Already Submitted" 
                        : !orgProfile?.wallet_address
                        ? "Connect Claim Wallet"
                        : "Submit a Claim"}
                  </Button>
                  )}
                </div>

                {/* Recent Activity */}
                <div className="pt-4 border-t">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-4 h-4 bg-green-800 rounded"></div>
                    <p className="text-sm text-green-800 font-medium">
                      Meet our top donors!
                    </p>
                  </div>

                  {/* Top Donors List */}
                  {topDonors.length > 0 ? (
                    <div className="space-y-4">
                      {topDonors.map((donation) => {
                        const name = getDonorName(donation)
                        const initial = getDonorInitial(donation)
                        
                        return (
                          <div key={donation.id} className="flex items-center gap-4">
                            <div className="w-14 h-14 rounded-full bg-green-800 flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                              {initial}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-bold text-base text-gray-900 truncate">{name}</p>
                              <p className="text-sm font-semibold text-green-800 mt-1">
                                {formatCurrency(donation.amount)}
                              </p>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                     <p className="text-sm text-gray-500 text-center py-4">No donors yet :(</p>
                  )}

                  <div className="mt-4">
                    <Dialog open={isDonorsModalOpen} onOpenChange={setIsDonorsModalOpen}>
                      <DialogTrigger asChild>
                        <Button
                          variant="ghost"
                          className="w-full text-xs"
                        >
                          View all donors
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-lg flex flex-col max-h-[85vh]">
                        <DialogHeader className="flex-shrink-0 pb-4 border-b">
                          <DialogTitle className="text-2xl font-bold">All Donors ({donations.length})</DialogTitle>
                        </DialogHeader>
                        <div className="overflow-y-auto pr-2 mt-4 space-y-4 flex-1 min-h-0" style={{ maxHeight: donations.length > 4 ? '400px' : 'auto' }}>
                          {donations.length > 0 ? (
                            donations.map((donation) => {
                              const name = getDonorName(donation)
                              const initial = getDonorInitial(donation)
                              const timeAgo = format(new Date(donation.donated_at), 'MMM dd, yyyy')
                              
                              return (
                                <div key={donation.id} className="flex items-center gap-4 border-b border-gray-200 pb-4 last:border-0">
                                  <div className="w-14 h-14 rounded-full bg-green-800 flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                                    {initial}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="flex justify-between items-baseline gap-2">
                                      <p className="font-bold text-base text-gray-900 truncate">{name}</p>
                                      <p className="font-bold text-base text-green-800 whitespace-nowrap">{formatCurrency(donation.amount)}</p>
                                    </div>
                                    <p className="text-sm text-gray-500 mt-1">
                                      {timeAgo}
                                    </p>
                                    {donation.message && (
                                      <p className="text-sm text-gray-600 mt-2 italic leading-relaxed">"{donation.message}"</p>
                                    )}
                                  </div>
                                </div>
                              )
                            })
                          ) : (
                            <p className="text-center text-gray-500 py-8">No donations yet.</p>
                          )}
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Donation Modal */}
      {event?.vault_address && (
        <DonationModal
          open={isDonationModalOpen}
          onOpenChange={setIsDonationModalOpen}
          vaultAddress={event.vault_address}
          eventId={event.id}
          onDonationSuccess={() => {
            // Refresh event details to show updated donations
            fetchEventDetails({ silent: true })
          }}
        />
      )}

      {/* Submit Claim Modal */}
      {event?.vault_address && dbUser?.id && (
        <SubmitClaimModal
          open={isSubmitClaimModalOpen}
          onOpenChange={setIsSubmitClaimModalOpen}
          vaultAddress={event.vault_address}
          eventId={event.id}
          organizationId={dbUser.id}
          onSuccess={() => {
            fetchEventDetails({ silent: true })
          }}
        />
      )}

      {/* Vote on Claims Modal */}
      {!isOrganization && selectedClaimForVote && (
        <Dialog open={isVoteModalOpen} onOpenChange={(open) => {
          setIsVoteModalOpen(open)
          if (!open) setSelectedClaimForVote(null)
        }}>
          <DialogContent className="max-w-[95vw] w-full lg:max-w-7xl max-h-[85vh] flex flex-col p-6">
          <DialogHeader className="flex-shrink-0 pb-4 border-b">
              <DialogTitle className="text-3xl font-bold text-gray-900">Vote on Claim</DialogTitle>
          </DialogHeader>
            <div className="flex-1 overflow-y-auto pr-2 mt-4 min-h-0">
              <Card className="border-0 shadow-none">
                <CardHeader className="pb-4 px-0">
                  <div className="flex flex-col md:flex-row justify-between items-start mb-6 gap-4">
                    <div className="flex-1">
                      <h3 className="text-2xl md:text-3xl font-bold text-gray-900 mb-3">{selectedClaimForVote.title}</h3>
                      <div className="flex items-center gap-3">
                        <span className="text-sm text-gray-500 uppercase tracking-wide font-medium">by</span>
                        {selectedClaimForVote.organization.organization_profiles?.profile_pic_url && (
                                <img 
                            src={selectedClaimForVote.organization.organization_profiles.profile_pic_url} 
                            alt={selectedClaimForVote.organization.organization_profiles.name}
                            className="w-8 h-8 rounded-full object-cover"
                                />
                              )}
                        <span className="text-base font-semibold text-gray-800">
                          {selectedClaimForVote.organization.organization_profiles?.name || 'Unknown Organization'}
                              </span>
                            </div>
                          </div>
                    <div className="flex flex-col items-end gap-2">
                      {getStatusBadge(selectedClaimForVote.status)}
                      <div className="text-right mt-2">
                        <span className="text-sm text-gray-500 uppercase tracking-wide font-semibold block mb-1">Allocated Relief Amount</span>
                        <span className="text-3xl font-bold text-green-800">{formatCurrency(selectedClaimForVote.requested_amount)}</span>
                      </div>
                    </div>
                        </div>
                      </CardHeader>
                <CardContent className="space-y-8 px-0">
                          <div>
                    <span className="text-sm text-gray-500 uppercase tracking-wide font-bold block mb-3">Description</span>
                    <p className="text-lg text-gray-700 leading-relaxed break-words whitespace-pre-wrap">{selectedClaimForVote.description}</p>
                          </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-8 border-t-2 border-gray-100">
                    {/* Vote Counts Section */}
                            <div>
                      <h4 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                        <span className="w-1 h-6 bg-gray-900 rounded-full"></span>
                        Current Vote Counts
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-green-50 rounded-xl p-5 border-2 border-green-100 text-center hover:border-green-200 transition-colors">
                          <div className="text-4xl font-extrabold text-green-800 mb-2">{selectedClaimForVote.accept_votes}</div>
                          <div className="text-sm font-bold text-green-700 uppercase tracking-wider">Accept</div>
                            </div>
                        <div className="bg-red-50 rounded-xl p-5 border-2 border-red-100 text-center hover:border-red-200 transition-colors">
                          <div className="text-4xl font-extrabold text-red-800 mb-2">{selectedClaimForVote.reject_votes}</div>
                          <div className="text-sm font-bold text-red-700 uppercase tracking-wider">Reject</div>
                            </div>
                        <div className="bg-blue-50 rounded-xl p-5 border-2 border-blue-100 text-center hover:border-blue-200 transition-colors">
                          <div className="text-4xl font-extrabold text-blue-800 mb-2">{selectedClaimForVote.raise_amount_votes}</div>
                          <div className="text-sm font-bold text-blue-700 uppercase tracking-wider">Raise Amount</div>
                            </div>
                        <div className="bg-orange-50 rounded-xl p-5 border-2 border-orange-100 text-center hover:border-orange-200 transition-colors">
                          <div className="text-4xl font-extrabold text-orange-800 mb-2">{selectedClaimForVote.lower_amount_votes}</div>
                          <div className="text-sm font-bold text-orange-700 uppercase tracking-wider">Lower Amount</div>
                            </div>
                          </div>
                        </div>
                    
                    {/* Vote Buttons Section */}
                    <div>
                      <h4 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                        <span className="w-1 h-6 bg-green-600 rounded-full"></span>
                        Cast Your Vote
                      </h4>
                      <div className="grid grid-cols-2 gap-4 h-[calc(100%-3.5rem)]">
                          <Button
                          variant={userVotes[selectedClaimForVote.id] === 'accept' ? 'default' : 'outline'}
                          className={`h-auto min-h-[100px] flex flex-col items-center justify-center gap-2 text-lg font-bold transition-all duration-200 ${
                            userVotes[selectedClaimForVote.id] === 'accept' 
                              ? 'bg-green-800 hover:bg-green-900 text-white shadow-lg scale-[1.02]' 
                              : 'border-2 border-green-200 bg-green-50/30 text-green-800 hover:bg-green-100 hover:border-green-300'
                          }`}
                          onClick={() => {
                            handleVote(selectedClaimForVote.id, 'accept')
                            setTimeout(() => {
                              setIsVoteModalOpen(false)
                              setSelectedClaimForVote(null)
                            }, 500)
                          }}
                          >
                          <CheckCircle className="w-8 h-8 mb-1" />
                          <span>Accept</span>
                          </Button>
                          <Button
                          variant={userVotes[selectedClaimForVote.id] === 'reject' ? 'default' : 'outline'}
                          className={`h-auto min-h-[100px] flex flex-col items-center justify-center gap-2 text-lg font-bold transition-all duration-200 ${
                            userVotes[selectedClaimForVote.id] === 'reject' 
                              ? 'bg-red-800 hover:bg-red-900 text-white shadow-lg scale-[1.02]' 
                              : 'border-2 border-red-200 bg-red-50/30 text-red-800 hover:bg-red-100 hover:border-red-300'
                          }`}
                          onClick={() => {
                            handleVote(selectedClaimForVote.id, 'reject')
                            setTimeout(() => {
                              setIsVoteModalOpen(false)
                              setSelectedClaimForVote(null)
                            }, 500)
                          }}
                          >
                          <XCircle className="w-8 h-8 mb-1" />
                          <span>Reject</span>
                          </Button>
                          <Button
                          variant={userVotes[selectedClaimForVote.id] === 'raise_amount' ? 'default' : 'outline'}
                          className={`h-auto min-h-[100px] flex flex-col items-center justify-center gap-2 text-lg font-bold transition-all duration-200 ${
                            userVotes[selectedClaimForVote.id] === 'raise_amount' 
                              ? 'bg-blue-800 hover:bg-blue-900 text-white shadow-lg scale-[1.02]' 
                              : 'border-2 border-blue-200 bg-blue-50/30 text-blue-800 hover:bg-blue-100 hover:border-blue-300'
                          }`}
                          onClick={() => {
                            handleVote(selectedClaimForVote.id, 'raise_amount')
                            setTimeout(() => {
                              setIsVoteModalOpen(false)
                              setSelectedClaimForVote(null)
                            }, 500)
                          }}
                          >
                          <TrendingUp className="w-8 h-8 mb-1" />
                          <span>Raise Amount</span>
                          </Button>
                          <Button
                          variant={userVotes[selectedClaimForVote.id] === 'lower_amount' ? 'default' : 'outline'}
                          className={`h-auto min-h-[100px] flex flex-col items-center justify-center gap-2 text-lg font-bold transition-all duration-200 ${
                            userVotes[selectedClaimForVote.id] === 'lower_amount' 
                              ? 'bg-orange-800 hover:bg-orange-900 text-white shadow-lg scale-[1.02]' 
                              : 'border-2 border-orange-200 bg-orange-50/30 text-orange-800 hover:bg-orange-100 hover:border-orange-300'
                          }`}
                          onClick={() => {
                            handleVote(selectedClaimForVote.id, 'lower_amount')
                            setTimeout(() => {
                              setIsVoteModalOpen(false)
                              setSelectedClaimForVote(null)
                            }, 500)
                          }}
                          >
                          <TrendingDown className="w-8 h-8 mb-1" />
                          <span>Lower Amount</span>
                          </Button>
                      </div>
                    </div>
                        </div>
                      </CardContent>
                    </Card>
          </div>
        </DialogContent>
      </Dialog>
      )}
    </div>
  )
}
