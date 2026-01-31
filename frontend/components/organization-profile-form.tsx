"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { Button } from "@/frontend/components/ui/button"
import { Input } from "@/frontend/components/ui/input"
import { Textarea } from "@/frontend/components/ui/textarea"
import { Label } from "@/frontend/components/ui/label"
import { useAuth } from "@/frontend/lib/auth"
import { Loader2, Upload, Building2, ArrowRight, ArrowLeft, Camera } from "lucide-react"
import { toast } from "@/frontend/hooks/use-toast"
import { Progress } from "@/frontend/components/ui/progress"
import { cn } from "@/frontend/lib/utils"

const organizationProfileSchema = z.object({
  name: z.string().min(1, "Organization name is required").trim(),
  location: z.string().trim().optional(),
  registration_number: z.string().trim().optional(),
  website_url: z.string().refine(
    (val) => !val || val === "" || z.string().url().safeParse(val).success,
    { message: "Invalid URL format" }
  ).optional(),
  phone: z.string().trim().optional(),
  description: z.string().trim().optional(),
  established_date: z.string().optional(),
})

type OrganizationProfileFormData = z.infer<typeof organizationProfileSchema>

const STEPS = [
  { id: 'identity', title: 'Organization Identity', fields: ['name'] },
  { id: 'details', title: 'Registration & Web', fields: ['registration_number', 'website_url'] },
  { id: 'contact', title: 'Contact & Location', fields: ['phone', 'location', 'established_date'] },
  { id: 'about', title: 'About Organization', fields: ['description'] },
]

export default function OrganizationProfileForm({ onComplete }: { onComplete: () => void }) {
  const { user, dbUser, syncUser } = useAuth()
  const [profilePic, setProfilePic] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [currentStep, setCurrentStep] = useState(0)

  const {
    register,
    handleSubmit,
    trigger,
    formState: { errors, isSubmitting },
  } = useForm<OrganizationProfileFormData>({
    resolver: zodResolver(organizationProfileSchema),
    mode: "onChange",
    defaultValues: {
      name: "",
      location: "",
      registration_number: "",
      website_url: "",
      phone: "",
      description: "",
      established_date: "",
    },
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setProfilePic(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const nextStep = async () => {
    const fields = STEPS[currentStep].fields as Array<keyof OrganizationProfileFormData>
    const isStepValid = await trigger(fields)
    if (isStepValid) {
      setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1))
    }
  }

  const prevStep = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0))
  }

  const onSubmit = async (data: OrganizationProfileFormData) => {
    if (!dbUser?.id) {
      toast({
        title: "Error",
        description: "User not found. Please try logging in again.",
        variant: "destructive",
      })
      return
    }

    setUploading(true)
    try {
      let profilePicUrl: string | null = null

      // Upload profile picture if provided
      if (profilePic) {
        const formData = new FormData()
        formData.append('file', profilePic)
        formData.append('userId', dbUser.id)
        formData.append('type', 'organization')

        const uploadResponse = await fetch('/api/upload-profile-pic', {
          method: 'POST',
          body: formData,
        })

        if (!uploadResponse.ok) {
          throw new Error('Failed to upload profile picture')
        }

        const uploadData = await uploadResponse.json()
        profilePicUrl = uploadData.url
      }

      // Create organization profile
      const response = await fetch('/api/create-profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          userType: 'organization',
          profile: {
            name: data.name.trim(),
            location: data.location?.trim() || null,
            registration_number: data.registration_number?.trim() || null,
            website_url: data.website_url?.trim() || null,
            phone: data.phone?.trim() || null,
            description: data.description?.trim() || null,
            established_date: data.established_date || null,
            profile_pic_url: profilePicUrl,
          },
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Failed to create profile')
      }

      // Update user type in users table
      await fetch('/api/update-user-type', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          userType: 'organization',
        }),
      })

      toast({
        title: "Success",
        description: "Organization profile created successfully!",
      })

      await syncUser()
      setTimeout(() => {
        onComplete()
      }, 500)
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to create profile. Please try again.",
        variant: "destructive",
      })
    } finally {
      setUploading(false)
    }
  }

  const progress = ((currentStep + 1) / STEPS.length) * 100

  const handleFormKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
    // Always prevent Enter key from submitting the form
    if (e.key === 'Enter') {
      e.preventDefault()
      // If not on last step, go to next step
      if (currentStep < STEPS.length - 1) {
        nextStep()
      }
      // If on last step, do nothing - user must click the Complete button
    }
  }

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    e.stopPropagation()
    // Only submit if we're on the last step AND this was triggered by the submit button
    if (currentStep === STEPS.length - 1) {
      const formData = new FormData(e.currentTarget)
      // Validate that this was actually triggered by the submit button
      await handleSubmit(onSubmit)(e)
    }
  }

  const handleCompleteClick = async () => {
    // Explicitly trigger form submission only when Complete button is clicked
    const form = document.querySelector('form') as HTMLFormElement
    if (form && currentStep === STEPS.length - 1) {
      const isValid = await trigger(STEPS[currentStep].fields as Array<keyof OrganizationProfileFormData>)
      if (isValid) {
        handleSubmit(onSubmit)()
      }
    }
  }

  return (
    <div className="w-full max-w-lg mx-auto">
      <div className="mb-8">
        <div className="flex justify-between text-sm font-medium text-gray-500 mb-2">
          <span>Step {currentStep + 1} of {STEPS.length}</span>
          <span>{STEPS[currentStep].title}</span>
        </div>
        <Progress value={progress} className="h-2 bg-gray-100" />
      </div>

      <form 
        onSubmit={handleFormSubmit}
        onKeyDown={handleFormKeyDown}
        className="space-y-6"
        noValidate
      >
        {/* Step 1: Identity */}
        <div className={cn("space-y-6", currentStep !== 0 && "hidden")}>
          <div className="flex flex-col items-center justify-center mb-6">
            <div className="relative group cursor-pointer">
              <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-white shadow-lg bg-gray-100 flex items-center justify-center">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <Building2 className="w-12 h-12 text-gray-400" />
                )}
              </div>
              <div className="absolute inset-0 rounded-full bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <Camera className="w-8 h-8 text-white" />
              </div>
              <Input
                id="profilePic"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>
            <p className="mt-3 text-sm text-gray-500 font-medium">Upload Organization Logo</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="name" className="text-base">Organization Name <span className="text-red-500">*</span></Label>
            <Input
              id="name"
              {...register("name")}
              placeholder="e.g. Acme Corp"
              autoComplete="organization"
              className="h-12 text-lg"
            />
            {errors.name && (
              <p className="text-sm text-destructive">{errors.name.message}</p>
            )}
          </div>
        </div>

        {/* Step 2: Registration & Web */}
        <div className={cn("space-y-6", currentStep !== 1 && "hidden")}>
          <div className="space-y-2">
            <Label htmlFor="registration_number" className="text-base">Registration Number</Label>
            <Input
              id="registration_number"
              {...register("registration_number")}
              placeholder="e.g. REG-123456"
              className="h-12"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="website_url" className="text-base">Website URL</Label>
            <Input
              id="website_url"
              type="url"
              {...register("website_url")}
              placeholder="https://example.com"
              className="h-12"
            />
            {errors.website_url && (
              <p className="text-sm text-destructive">{errors.website_url.message}</p>
            )}
          </div>
        </div>

        {/* Step 3: Contact & Location */}
        <div className={cn("space-y-6", currentStep !== 2 && "hidden")}>
          <div className="space-y-2">
            <Label htmlFor="phone" className="text-base">Phone Number</Label>
            <Input
              id="phone"
              {...register("phone")}
              placeholder="e.g. +1 (555) 000-0000"
              className="h-12"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="location" className="text-base">Location</Label>
            <Input
              id="location"
              {...register("location")}
              placeholder="e.g. San Francisco, CA"
              className="h-12"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="established_date" className="text-base">Established Date</Label>
            <Input
              id="established_date"
              type="date"
              {...register("established_date")}
              className="h-12"
            />
          </div>
        </div>

        {/* Step 4: About */}
        <div className={cn("space-y-6", currentStep !== 3 && "hidden")}>
          <div className="space-y-2">
            <Label htmlFor="description" className="text-base">Description</Label>
            <Textarea
              id="description"
              {...register("description")}
              placeholder="Tell us about your organization's mission and goals..."
              rows={6}
              className="resize-none text-base p-4"
            />
            <p className="text-xs text-gray-400 text-right">Optional</p>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between pt-6 border-t border-gray-100 mt-8">
          <Button
            type="button"
            variant="ghost"
            onClick={prevStep}
            disabled={currentStep === 0}
            className={cn(currentStep === 0 && "invisible")}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          {currentStep < STEPS.length - 1 ? (
            <Button
              type="button"
              onClick={nextStep}
              className="bg-green-800 hover:bg-green-900 min-w-[120px]"
            >
              Next
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          ) : (
            <Button
              type="button"
              onClick={handleCompleteClick}
              className="bg-green-800 hover:bg-green-900 min-w-[120px]"
              disabled={isSubmitting || uploading}
            >
              {(isSubmitting || uploading) ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  Complete
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          )}
        </div>
      </form>
    </div>
  )
}
