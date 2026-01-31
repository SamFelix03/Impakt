"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuthGuard } from "@/frontend/lib/auth-guard"
import { useAuth } from "@/frontend/lib/auth"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/frontend/components/ui/card"
import { Button } from "@/frontend/components/ui/button"
import { Input } from "@/frontend/components/ui/input"
import { Textarea } from "@/frontend/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/frontend/components/ui/tabs"
import { Loader2, User, Building2, MapPin, Phone, Globe, Calendar, Edit2, Check, X, DollarSign, AlertCircle, FileText, Wallet, LogOut } from "lucide-react"
import { Badge } from "@/frontend/components/ui/badge"
import Navbar from "@/frontend/components/navbar"
import { format } from "date-fns"
import { toast } from "@/frontend/hooks/use-toast"
import { truncateAddress } from "@/frontend/lib/utils"

interface Donation {
  id: string
  amount: number
  donated_at: string
  disaster_event: {
    id: string
    title: string
    location: string
    disaster_type: string | null
    severity: string | null
  } | null
}

interface Claim {
  id: string
  title: string
  description: string
  requested_amount: number
  status: 'pending' | 'accepted' | 'rejected' | 'raise_amount' | 'lower_amount'
  created_at: string
  updated_at: string
  disaster_event: {
    id: string
    title: string
    location: string
    disaster_type: string | null
    severity: string | null
  } | null
}

interface EditableFieldProps {
  label: string
  value: string | number | null | undefined
  isEditing: boolean
  onEdit: () => void
  onSave: (value: string | number | null) => void
  onCancel: () => void
  type?: 'text' | 'number' | 'textarea' | 'date' | 'url'
  icon?: React.ReactNode
  placeholder?: string
}

function EditableField({ 
  label, 
  value, 
  isEditing, 
  onEdit, 
  onSave, 
  onCancel,
  type = 'text',
  icon,
  placeholder
}: EditableFieldProps) {
  const [editValue, setEditValue] = useState(value?.toString() || '')

  useEffect(() => {
    setEditValue(value?.toString() || '')
  }, [value, isEditing])

  const handleSave = () => {
    if (type === 'number') {
      const numValue = editValue ? parseInt(editValue, 10) : null
      onSave(numValue)
    } else {
      onSave(editValue || null)
    }
  }

  return (
    <div className="flex items-start gap-4 py-4 border-b border-gray-100 last:border-0">
      <div className="flex-shrink-0 mt-1">
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-500 mb-2 uppercase tracking-wide">{label}</p>
        {isEditing ? (
          <div className="space-y-2">
            {type === 'textarea' ? (
              <Textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                placeholder={placeholder}
                className="min-h-24"
              />
            ) : (
              <Input
                type={type}
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                placeholder={placeholder}
                className="text-lg"
              />
            )}
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={handleSave}
                className="bg-green-800 hover:bg-green-900"
              >
                <Check className="w-4 h-4 mr-1" />
                Save
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={onCancel}
              >
                <X className="w-4 h-4 mr-1" />
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between group">
            <p className="text-xl font-semibold text-gray-900 break-words">
              {value || <span className="text-gray-400 italic">Not set</span>}
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={onEdit}
              className="opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Edit2 className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

function DonationCard({ donation }: { donation: Donation }) {
  const router = useRouter()
  const event = donation.disaster_event

  return (
    <Card className="hover:shadow-lg transition-all duration-300 border-l-4 border-l-green-800">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-2xl font-bold text-gray-900 mb-3">
              {event?.title || 'Unknown Event'}
            </CardTitle>
            {event && (
              <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600">
                {event.location && (
                  <div className="flex items-center gap-1.5">
                    <MapPin className="w-4 h-4" />
                    <span className="font-medium">{event.location}</span>
                  </div>
                )}
                {event.disaster_type && (
                  <span className="px-3 py-1 bg-gray-100 rounded-full uppercase text-xs font-semibold tracking-wide">
                    {event.disaster_type}
                  </span>
                )}
                {event.severity && (
                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                    event.severity === 'high' ? 'bg-red-100 text-red-800' :
                    event.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {event.severity}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-gray-100">
            <span className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Amount Donated</span>
            <span className="text-3xl font-bold text-green-800">
              ${donation.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex items-center gap-2 text-base text-gray-600">
            <Calendar className="w-5 h-5" />
            <span className="font-medium">{format(new Date(donation.donated_at), 'MMMM dd, yyyy')}</span>
          </div>
          {event && (
            <Button
              variant="outline"
              className="w-full mt-6 font-semibold"
              onClick={() => router.push(`/event/${event.id}`)}
            >
              View Event
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function ClaimCard({ claim }: { claim: Claim }) {
  const router = useRouter()
  const event = claim.disaster_event

  const getStatusBadge = (status: Claim['status']) => {
    switch (status) {
      case 'accepted':
        return <Badge className="bg-green-100 text-green-800 hover:bg-green-200 border-green-200 font-semibold">Accepted</Badge>
      case 'rejected':
        return <Badge className="bg-red-100 text-red-800 hover:bg-red-200 border-red-200 font-semibold">Rejected</Badge>
      case 'pending':
        return <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-200 border-yellow-200 font-semibold">Pending</Badge>
      case 'raise_amount':
        return <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-200 border-blue-200 font-semibold">Raise Amount</Badge>
      case 'lower_amount':
        return <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-200 border-orange-200 font-semibold">Lower Amount</Badge>
      default:
        return <Badge variant="outline">Unknown</Badge>
    }
  }

  return (
    <Card className="hover:shadow-lg transition-all duration-300 border-l-4 border-l-green-800">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-2xl font-bold text-gray-900 mb-3">
              {event?.title || 'Unknown Event'}
            </CardTitle>
            {event && (
              <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600">
                <div className="flex items-center gap-1.5">
                  <MapPin className="w-4 h-4" />
                  <span className="font-medium">{event.location}</span>
                </div>
                {event.disaster_type && (
                  <span className="px-3 py-1 bg-gray-100 rounded-full uppercase text-xs font-semibold tracking-wide">
                    {event.disaster_type}
                  </span>
                )}
                {event.severity && (
                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                    event.severity === 'high' ? 'bg-red-100 text-red-800' :
                    event.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {event.severity}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4 mb-3">
            <CardTitle className="text-xl font-bold text-gray-900 flex-1">
              {claim.title}
            </CardTitle>
            {getStatusBadge(claim.status)}
          </div>
          <div className="flex items-center justify-between py-3 border-b border-gray-100">
            <span className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Allocated Relief Amount</span>
            <span className="text-3xl font-bold text-green-800">
              ${claim.requested_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          {claim.description && (
            <p className="text-base text-gray-700 leading-relaxed line-clamp-3 font-medium">
              {claim.description}
            </p>
          )}
          <div className="flex items-center gap-2 text-base text-gray-600">
            <Calendar className="w-5 h-5" />
            <span className="font-medium">Submitted {format(new Date(claim.created_at), 'MMMM dd, yyyy')}</span>
          </div>
          {event && (
            <Button
              variant="outline"
              className="w-full mt-6 font-semibold"
              onClick={() => router.push(`/event/${event.id}`)}
            >
              View Event
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function DashboardPage() {
  const router = useRouter()
  const { isAuthorized, isLoading: authLoading } = useAuthGuard({
    requireAuth: true,
    requireProfile: true,
  })
  const { userProfile, orgProfile, dbUser, syncUser } = useAuth()
  const isOrganization = !!orgProfile
  const [donations, setDonations] = useState<Donation[]>([])
  const [donationsLoading, setDonationsLoading] = useState(true)
  const [donationsError, setDonationsError] = useState<string | null>(null)
  const [claims, setClaims] = useState<Claim[]>([])
  const [claimsLoading, setClaimsLoading] = useState(true)
  const [claimsError, setClaimsError] = useState<string | null>(null)
  const [editingField, setEditingField] = useState<string | null>(null)
  const [profileData, setProfileData] = useState<any>(null)
  const [uploadingPic, setUploadingPic] = useState(false)
  const [walletAddress, setWalletAddress] = useState<string>("")
  const [savingWallet, setSavingWallet] = useState(false)

  useEffect(() => {
    if (isAuthorized && dbUser) {
      if (isOrganization) {
        fetchClaims()
      } else {
        fetchDonations()
      }
      const profile = userProfile || orgProfile
      if (profile) {
        setProfileData({ ...profile })
        if (isOrganization && profile.wallet_address) {
          setWalletAddress(profile.wallet_address)
        }
      }
    }
  }, [isAuthorized, dbUser, userProfile, orgProfile, isOrganization])

  const fetchDonations = async () => {
    if (!dbUser?.id) return

    try {
      setDonationsLoading(true)
      setDonationsError(null)
      
      const response = await fetch(`/api/my-donations?userId=${dbUser.id}`)
      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch donations')
      }

      setDonations(result.data || [])
    } catch (err: any) {
      console.error('Error fetching donations:', err)
      setDonationsError(err.message || 'Failed to load donations')
    } finally {
      setDonationsLoading(false)
    }
  }

  const fetchClaims = async () => {
    if (!dbUser?.id) return

    try {
      setClaimsLoading(true)
      setClaimsError(null)
      
      const response = await fetch(`/api/my-claims?organizationId=${dbUser.id}`)
      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch claims')
      }

      setClaims(result.data || [])
    } catch (err: any) {
      console.error('Error fetching claims:', err)
      setClaimsError(err.message || 'Failed to load claims')
    } finally {
      setClaimsLoading(false)
    }
  }

  const handleFieldEdit = (field: string) => {
    setEditingField(field)
  }

  const handleFieldCancel = () => {
    setEditingField(null)
    // Reset to original value
    const profile = userProfile || orgProfile
    if (profile) {
      setProfileData({ ...profile })
    }
  }

  const handleFieldSave = async (field: string, value: string | number | null) => {
    if (!dbUser?.id || !profileData) return

    try {
      const updatedProfile = { ...profileData, [field]: value }

      const response = await fetch('/api/update-profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          userType: isOrganization ? 'organization' : 'user',
          profile: updatedProfile,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to update profile')
      }

      setProfileData(updatedProfile)
      setEditingField(null)
      await syncUser() // Refresh auth state
      
      toast({
        title: "Success",
        description: "Profile updated successfully",
      })
    } catch (err: any) {
      console.error('Error updating profile:', err)
      toast({
        title: "Error",
        description: err.message || 'Failed to update profile',
        variant: "destructive",
      })
    }
  }

  const handleProfilePicChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !dbUser?.id) return

    setUploadingPic(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('userId', dbUser.id)
      formData.append('type', orgProfile ? 'organization' : 'user')

      const uploadResponse = await fetch('/api/upload-profile-pic', {
        method: 'POST',
        body: formData,
      })

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload profile picture')
      }

      const uploadData = await uploadResponse.json()
      await handleFieldSave('profile_pic_url', uploadData.url)
    } catch (err: any) {
      console.error('Error uploading profile picture:', err)
      toast({
        title: "Error",
        description: err.message || 'Failed to upload profile picture',
        variant: "destructive",
      })
    } finally {
      setUploadingPic(false)
    }
  }

  const handleSaveWalletAddress = async () => {
    if (!dbUser?.id || !isOrganization) return

    // Basic validation
    if (!walletAddress.trim()) {
      toast({
        title: "Error",
        description: "Please enter a wallet address",
        variant: "destructive",
      })
      return
    }

    // Basic Ethereum address validation
    if (!/^0x[a-fA-F0-9]{40}$/.test(walletAddress.trim())) {
      toast({
        title: "Error",
        description: "Please enter a valid Ethereum wallet address (0x followed by 40 hex characters)",
        variant: "destructive",
      })
      return
    }

    try {
      setSavingWallet(true)
      
      const response = await fetch('/api/update-wallet-address', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          walletAddress: walletAddress.trim(),
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to save wallet address')
      }

      await syncUser() // Refresh auth state
      setEditingField(null)
      
      toast({
        title: "Success",
        description: "Wallet address saved successfully",
      })
    } catch (err: any) {
      console.error('Error saving wallet address:', err)
      toast({
        title: "Error",
        description: err.message || 'Failed to save wallet address',
        variant: "destructive",
      })
    } finally {
      setSavingWallet(false)
    }
  }

  const handleClearWalletAddress = async () => {
    if (!dbUser?.id || !isOrganization) return

    try {
      setSavingWallet(true)
      
      const response = await fetch('/api/update-wallet-address', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          walletAddress: null,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to clear wallet address')
      }

      await syncUser() // Refresh auth state
      setWalletAddress("")
      setEditingField(null)
      
      toast({
        title: "Success",
        description: "Wallet address cleared successfully",
      })
    } catch (err: any) {
      console.error('Error clearing wallet address:', err)
      toast({
        title: "Error",
        description: err.message || 'Failed to clear wallet address',
        variant: "destructive",
      })
    } finally {
      setSavingWallet(false)
    }
  }

  if (authLoading || !isAuthorized) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-green-800" />
      </div>
    )
  }

  const profile = userProfile || orgProfile

  if (!profile || !dbUser) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-500">Profile not found</p>
      </div>
    )
  }

  const currentProfileData = profileData || profile

  return (
    <div className="min-h-screen bg-gray-50 pt-28 pb-8">
      <Navbar />
      
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-16 mt-8 text-center">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-4 tracking-tight">Dashboard</h1>
          <p className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto">Manage your profile and view your donation history</p>
        </div>

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 mb-8">
            <TabsTrigger value="profile">My Profile</TabsTrigger>
            <TabsTrigger value={isOrganization ? "claims" : "donations"}>
              {isOrganization ? "My Claims" : "My Donations"}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="space-y-6">
          {/* Profile Header Card */}
          <Card>
            <CardHeader>
                <div className="flex flex-col items-center gap-6">
                  <div className="relative">
                    {currentProfileData.profile_pic_url ? (
                  <img
                        src={currentProfileData.profile_pic_url}
                        alt={currentProfileData.name}
                        className="w-32 h-32 rounded-full object-cover border-4 border-green-800"
                  />
                ) : (
                      <div className="w-32 h-32 rounded-full bg-green-800 flex items-center justify-center">
                    {isOrganization ? (
                          <Building2 className="w-16 h-16 text-white" />
                    ) : (
                          <User className="w-16 h-16 text-white" />
                        )}
                      </div>
                    )}
                    <label className="absolute bottom-0 right-0 bg-green-800 text-white rounded-full p-2 cursor-pointer hover:bg-green-900 transition-colors">
                      {uploadingPic ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Edit2 className="w-4 h-4" />
                    )}
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleProfilePicChange}
                        className="hidden"
                        disabled={uploadingPic}
                      />
                    </label>
                  </div>
                  <div className="text-center">
                    <CardTitle className="text-3xl text-gray-900 mb-2">{currentProfileData.name}</CardTitle>
                  {isOrganization && orgProfile.registration_number && (
                      <CardDescription className="text-base">
                      Registration: {orgProfile.registration_number}
                    </CardDescription>
                  )}
                </div>
                  {isOrganization && (
                    <div className="w-full max-w-md pt-4 border-t border-gray-100">
                      <p className="text-sm font-medium text-gray-500 mb-3 uppercase tracking-wide text-center">Claim Wallet</p>
                      {editingField === 'wallet_address' ? (
                        <div className="space-y-3">
                          <Input
                            type="text"
                            value={walletAddress}
                            onChange={(e) => setWalletAddress(e.target.value)}
                            placeholder="0x..."
                            className="font-mono text-base"
                            disabled={savingWallet}
                          />
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={handleSaveWalletAddress}
                              disabled={savingWallet}
                              className="bg-green-800 hover:bg-green-900 flex-1"
                            >
                              {savingWallet ? (
                                <>
                                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                  Saving...
                                </>
                              ) : (
                                <>
                                  <Check className="w-4 h-4 mr-2" />
                                  Save
                                </>
                              )}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                setEditingField(null)
                                setWalletAddress(currentProfileData.wallet_address || "")
                              }}
                              disabled={savingWallet}
                            >
                              <X className="w-4 h-4 mr-2" />
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-3">
                          {currentProfileData.wallet_address ? (
                            <>
                              <div className="flex items-center gap-2 w-full justify-center">
                                <Wallet className="w-5 h-5 text-green-800 flex-shrink-0" />
                                <p className="text-lg font-semibold text-gray-900 font-mono">
                                  {truncateAddress(currentProfileData.wallet_address)}
                                </p>
                              </div>
                              <div className="flex gap-2 w-full">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => {
                                    setEditingField('wallet_address')
                                    setWalletAddress(currentProfileData.wallet_address)
                                  }}
                                  className="flex-1"
                                >
                                  <Edit2 className="w-4 h-4 mr-2" />
                                  Edit
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={handleClearWalletAddress}
                                  disabled={savingWallet}
                                  className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                >
                                  {savingWallet ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                  ) : (
                                    <X className="w-4 h-4" />
                                  )}
                                </Button>
                              </div>
                            </>
                          ) : (
                            <>
                              <Button
                                onClick={() => setEditingField('wallet_address')}
                                className="bg-green-800 hover:bg-green-900 text-white font-semibold w-full"
                              >
                                <Wallet className="w-4 h-4 mr-2" />
                                Connect Wallet
                              </Button>
                              <p className="text-xs text-gray-500 text-center">
                                Wallet address is required to submit claims
                              </p>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}
              </div>
            </CardHeader>
          </Card>

          {/* Profile Details Card */}
          <Card>
            <CardHeader>
                <CardTitle className="text-2xl text-gray-900">Profile Details</CardTitle>
            </CardHeader>
              <CardContent className="space-y-0">
                <EditableField
                  label="Name"
                  value={currentProfileData.name}
                  isEditing={editingField === 'name'}
                  onEdit={() => handleFieldEdit('name')}
                  onSave={(value) => handleFieldSave('name', value)}
                  onCancel={handleFieldCancel}
                  icon={<User className="w-5 h-5" />}
                  placeholder="Enter your name"
                />

              {!isOrganization && userProfile && (
                  <EditableField
                    label="Age"
                    value={currentProfileData.age}
                    isEditing={editingField === 'age'}
                    onEdit={() => handleFieldEdit('age')}
                    onSave={(value) => handleFieldSave('age', value)}
                    onCancel={handleFieldCancel}
                    type="number"
                    icon={<User className="w-5 h-5" />}
                    placeholder="Enter your age"
                  />
                )}

                <EditableField
                  label="Location"
                  value={currentProfileData.location}
                  isEditing={editingField === 'location'}
                  onEdit={() => handleFieldEdit('location')}
                  onSave={(value) => handleFieldSave('location', value)}
                  onCancel={handleFieldCancel}
                  icon={<MapPin className="w-5 h-5" />}
                  placeholder="Enter your location"
                />

                <EditableField
                  label="Phone"
                  value={currentProfileData.phone}
                  isEditing={editingField === 'phone'}
                  onEdit={() => handleFieldEdit('phone')}
                  onSave={(value) => handleFieldSave('phone', value)}
                  onCancel={handleFieldCancel}
                  icon={<Phone className="w-5 h-5" />}
                  placeholder="Enter your phone number"
                />

                {isOrganization && (
                  <>
                    <EditableField
                      label="Website"
                      value={currentProfileData.website_url}
                      isEditing={editingField === 'website_url'}
                      onEdit={() => handleFieldEdit('website_url')}
                      onSave={(value) => handleFieldSave('website_url', value)}
                      onCancel={handleFieldCancel}
                      type="url"
                      icon={<Globe className="w-5 h-5" />}
                      placeholder="https://example.com"
                    />
                    <EditableField
                      label="Registration Number"
                      value={currentProfileData.registration_number}
                      isEditing={editingField === 'registration_number'}
                      onEdit={() => handleFieldEdit('registration_number')}
                      onSave={(value) => handleFieldSave('registration_number', value)}
                      onCancel={handleFieldCancel}
                      icon={<Building2 className="w-5 h-5" />}
                      placeholder="Enter registration number"
                    />
                    <EditableField
                      label="Established Date"
                      value={currentProfileData.established_date ? format(new Date(currentProfileData.established_date), 'yyyy-MM-dd') : null}
                      isEditing={editingField === 'established_date'}
                      onEdit={() => handleFieldEdit('established_date')}
                      onSave={(value) => handleFieldSave('established_date', value)}
                      onCancel={handleFieldCancel}
                      type="date"
                      icon={<Calendar className="w-5 h-5" />}
                    />
                </>
              )}
            </CardContent>
          </Card>

          {/* Bio/Description Card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl text-gray-900">{isOrganization ? 'About' : 'Bio'}</CardTitle>
              </CardHeader>
              <CardContent>
                <EditableField
                  label={isOrganization ? 'Description' : 'Bio'}
                  value={isOrganization ? currentProfileData.description : currentProfileData.bio}
                  isEditing={editingField === (isOrganization ? 'description' : 'bio')}
                  onEdit={() => handleFieldEdit(isOrganization ? 'description' : 'bio')}
                  onSave={(value) => handleFieldSave(isOrganization ? 'description' : 'bio', value)}
                  onCancel={handleFieldCancel}
                  type="textarea"
                  placeholder={`Enter your ${isOrganization ? 'description' : 'bio'}`}
                />
              </CardContent>
            </Card>

          {/* Account Info Card */}
          <Card>
            <CardHeader>
                <CardTitle className="text-2xl text-gray-900">Account Information</CardTitle>
            </CardHeader>
              <CardContent className="space-y-4">
                <div className="py-2">
                  <p className="text-sm font-medium text-gray-500 mb-2 uppercase tracking-wide">Email</p>
                  <p className="text-xl font-semibold text-gray-900">{dbUser.email}</p>
              </div>
                <div className="py-2">
                  <p className="text-sm font-medium text-gray-500 mb-2 uppercase tracking-wide">Account Type</p>
                  <p className="text-xl font-semibold text-gray-900 capitalize">
                  {isOrganization ? 'Organization' : 'Individual'}
                </p>
              </div>
            </CardContent>
          </Card>
          </TabsContent>

          {!isOrganization && (
            <TabsContent value="donations" className="space-y-6">
              {donationsLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="h-8 w-8 animate-spin text-green-800" />
                </div>
              ) : donationsError ? (
                <div className="flex flex-col items-center justify-center py-16">
                  <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
                  <p className="text-red-600 mb-4">{donationsError}</p>
                  <Button 
                    onClick={fetchDonations} 
                    className="bg-green-800 hover:bg-green-900"
                  >
                    Try Again
                  </Button>
                </div>
              ) : donations.length === 0 ? (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-16">
                    <DollarSign className="w-16 h-16 text-gray-300 mb-4" />
                    <p className="text-gray-500 text-lg mb-2">No donations yet</p>
                    <p className="text-gray-400 text-sm mb-6">Start making a difference by donating to disaster relief efforts</p>
                    <Button 
                      onClick={() => router.push('/home')}
                      className="bg-green-800 hover:bg-green-900"
                    >
                      Browse Events
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {donations.map((donation) => (
                    <DonationCard key={donation.id} donation={donation} />
                  ))}
                </div>
              )}
            </TabsContent>
          )}

          {isOrganization && (
            <TabsContent value="claims" className="space-y-6">
              {claimsLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="h-8 w-8 animate-spin text-green-800" />
                </div>
              ) : claimsError ? (
                <div className="flex flex-col items-center justify-center py-16">
                  <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
                  <p className="text-red-600 mb-4">{claimsError}</p>
                  <Button 
                    onClick={fetchClaims} 
                    className="bg-green-800 hover:bg-green-900"
                  >
                    Try Again
                  </Button>
                </div>
              ) : claims.length === 0 ? (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-16">
                    <FileText className="w-16 h-16 text-gray-300 mb-4" />
                    <p className="text-gray-500 text-lg mb-2">No claims yet</p>
                    <p className="text-gray-400 text-sm mb-6">Submit your first claim to request funds for disaster relief efforts</p>
                    <Button 
                      onClick={() => router.push('/home')}
                      className="bg-green-800 hover:bg-green-900"
                    >
                      Browse Events
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {claims.map((claim) => (
                    <ClaimCard key={claim.id} claim={claim} />
                  ))}
        </div>
              )}
            </TabsContent>
          )}
        </Tabs>
      </div>
    </div>
  )
}
