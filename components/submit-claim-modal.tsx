"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Loader2, Upload, CheckCircle, X, Image as ImageIcon } from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"
import { useAuth } from "@/lib/auth"

interface SubmitClaimModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  vaultAddress: string
  eventId: string
  organizationId: string
  onSuccess: () => void
}

interface ClaimResult {
  id: string
  title: string
  description: string
  requested_amount: number
  status: string
  created_at: string
}

export function SubmitClaimModal({
  open,
  onOpenChange,
  vaultAddress,
  eventId,
  organizationId,
  onSuccess,
}: SubmitClaimModalProps) {
  const { orgProfile } = useAuth()
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [images, setImages] = useState<File[]>([])
  const [imagePreviews, setImagePreviews] = useState<string[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submittedClaim, setSubmittedClaim] = useState<ClaimResult | null>(null)
  const [recommendedAmount, setRecommendedAmount] = useState<number | null>(null)

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setImages(files)
    
    // Create preview URLs
    const previews = files.map(file => URL.createObjectURL(file))
    setImagePreviews(previews)
  }

  const removeImage = (index: number) => {
    // Revoke the object URL to free memory
    if (imagePreviews[index]) {
      URL.revokeObjectURL(imagePreviews[index])
    }
    
    const newImages = images.filter((_, i) => i !== index)
    const newPreviews = imagePreviews.filter((_, i) => i !== index)
    
    setImages(newImages)
    setImagePreviews(newPreviews)
  }

  const handleSubmit = async () => {
    if (!title.trim() || !description.trim()) {
      toast({
        title: "Error",
        description: "Please fill in both title and description",
        variant: "destructive",
      })
      return
    }

    if (!orgProfile?.wallet_address) {
      toast({
        title: "Error",
        description: "Please connect your wallet in the dashboard before submitting a claim",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)
    try {
      const formData = new FormData()
      // Combine title and description as content (API expects content field)
      const content = `${title}\n\n${description}`
      formData.append('content', content)
      formData.append('vault_address', vaultAddress)
      formData.append('disaster_event_id', eventId)
      formData.append('organization_id', organizationId)
      
      // Add images
      images.forEach((image) => {
        formData.append('images', image)
      })

      const response = await fetch('/api/submit-claim', {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to submit claim')
      }

      setSubmittedClaim(result.claim)
      setRecommendedAmount(result.verification?.recommended_amount_usd || result.claim.requested_amount)
      
      toast({
        title: "Success",
        description: "Claim submitted successfully!",
      })

      // Refresh the claims list
      onSuccess()
    } catch (error: any) {
      console.error('Error submitting claim:', error)
      toast({
        title: "Error",
        description: error.message || "Failed to submit claim. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    if (!isSubmitting) {
      // Clean up preview URLs
      imagePreviews.forEach(preview => URL.revokeObjectURL(preview))
      
      setTitle("")
      setDescription("")
      setImages([])
      setImagePreviews([])
      setSubmittedClaim(null)
      setRecommendedAmount(null)
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-[90vw] md:max-w-5xl lg:max-w-6xl max-h-[85vh] overflow-y-auto w-full p-6 md:p-8">
        {!submittedClaim ? (
          <>
            <DialogHeader>
              <DialogTitle className="text-3xl font-bold text-gray-900">Submit a Claim</DialogTitle>
            </DialogHeader>

            <div className="space-y-8 mt-6">
              <div className="space-y-3">
                <Label htmlFor="title" className="text-lg font-semibold text-gray-700">
                  Claim Title <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Emergency Food Relief for Chile Wildfire Victims"
                  className="h-14 text-lg w-full"
                  disabled={isSubmitting}
                />
              </div>

              <div className="space-y-3">
                <Label htmlFor="description" className="text-lg font-semibold text-gray-700">
                  Description <span className="text-red-500">*</span>
                </Label>
                <Textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Provide detailed information about your claim, including the purpose, impact, and how the funds will be used..."
                  rows={12}
                  className="min-h-[300px] w-full text-lg p-5 leading-relaxed resize-y break-words whitespace-pre-wrap"
                  disabled={isSubmitting}
                />
              </div>

              <div className="space-y-3">
                <Label htmlFor="images" className="text-lg font-semibold text-gray-700">
                  Supporting Images (Optional)
                </Label>
                
                {/* Image Upload Area */}
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-green-500 transition-colors">
                  <label
                    htmlFor="images"
                    className={cn(
                      "flex flex-col items-center justify-center cursor-pointer",
                      isSubmitting && "cursor-not-allowed opacity-50"
                    )}
                  >
                    <div className="flex flex-col items-center gap-2">
                      <div className="w-12 h-12 bg-green-50 rounded-full flex items-center justify-center">
                        <ImageIcon className="w-6 h-6 text-green-800" />
                      </div>
                      <div className="text-center">
                        <span className="text-sm font-medium text-gray-700">
                          Click to upload images
                        </span>
                        <p className="text-xs text-gray-500 mt-1">
                          PNG, JPG, GIF up to 10MB each
                        </p>
                      </div>
                    </div>
                    <Input
                      id="images"
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handleImageChange}
                      className="hidden"
                      disabled={isSubmitting}
                    />
                  </label>
                </div>

                {/* Image Previews */}
                {imagePreviews.length > 0 && (
                  <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mt-4">
                    {imagePreviews.map((preview, index) => (
                      <div key={index} className="relative group">
                        <div className="aspect-square rounded-lg overflow-hidden border-2 border-gray-200 bg-gray-50">
                          <img
                            src={preview}
                            alt={`Preview ${index + 1}`}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => removeImage(index)}
                          className="absolute top-2 right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                          disabled={isSubmitting}
                        >
                          <X className="w-4 h-4" />
                        </button>
                        <div className="mt-1 text-xs text-gray-500 truncate">
                          {images[index]?.name || `Image ${index + 1}`}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                <p className="text-xs text-gray-400">
                  Upload images that support your claim (receipts, photos, documents, etc.)
                </p>
              </div>

              <div className="flex justify-end gap-3 pt-4 border-t">
                <Button
                  variant="outline"
                  onClick={handleClose}
                  disabled={isSubmitting}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmit}
                  className="bg-green-800 hover:bg-green-900 min-w-[120px]"
                  disabled={isSubmitting || !title.trim() || !description.trim()}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Submitting...
                    </>
                  ) : (
                    "Submit Claim"
                  )}
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
              <CheckCircle className="w-10 h-10 text-green-800" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">Claim Submitted Successfully!</h3>
            <p className="text-gray-600 mb-6">
              Your claim has been submitted and is now pending review.
            </p>

            <div className="w-full bg-gray-50 rounded-lg p-6 space-y-4 text-left">
              <div>
                <p className="text-sm text-gray-500 mb-1">Claim Title</p>
                <p className="font-semibold text-gray-900">{submittedClaim.title}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Allocated Relief Amount</p>
                <p className="text-2xl font-bold text-green-800">
                  ${recommendedAmount?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Status</p>
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800">
                  Pending Review
                </span>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Claim ID</p>
                <p className="font-mono text-xs text-gray-600">{submittedClaim.id}</p>
              </div>
            </div>

            <p className="text-sm text-gray-500 mt-6">
              Your claim will be reviewed by the community. You'll be notified once a decision is made.
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
