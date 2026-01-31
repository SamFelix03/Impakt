"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/lib/auth"
import { Loader2, Upload, User, ArrowRight, ArrowLeft, Camera } from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

const individualProfileSchema = z.object({
  name: z.string().min(1, "Name is required").trim(),
  age: z.preprocess(
    (val) => {
      if (val === "" || val === null || val === undefined) return undefined
      const num = typeof val === "string" ? parseInt(val, 10) : Number(val)
      return isNaN(num) ? undefined : num
    },
    z.number().int().positive().max(150, "Age must be between 1 and 150").optional()
  ),
  location: z.string().trim().optional(),
  phone: z.string().trim().optional(),
  bio: z.string().trim().optional(),
})

type IndividualProfileFormData = z.infer<typeof individualProfileSchema>

const STEPS = [
  { id: 'identity', title: 'Identity', fields: ['name'] },
  { id: 'details', title: 'Basic Details', fields: ['age', 'location', 'phone'] },
  { id: 'about', title: 'About You', fields: ['bio'] },
]

export default function IndividualProfileForm({ onComplete }: { onComplete: () => void }) {
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
  } = useForm<IndividualProfileFormData>({
    resolver: zodResolver(individualProfileSchema),
    mode: "onChange",
    defaultValues: {
      name: "",
      age: undefined,
      location: "",
      phone: "",
      bio: "",
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
    const fields = STEPS[currentStep].fields as Array<keyof IndividualProfileFormData>
    const isStepValid = await trigger(fields)
    if (isStepValid) {
      setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1))
    }
  }

  const prevStep = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0))
  }

  const onSubmit = async (data: IndividualProfileFormData) => {
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
        formData.append('type', 'user')

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

      // Create user profile
      const response = await fetch('/api/create-profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: dbUser.id,
          userType: 'user',
          profile: {
            name: data.name.trim(),
            age: data.age && typeof data.age === 'number' ? data.age : (data.age ? parseInt(String(data.age), 10) : null),
            location: data.location?.trim() || null,
            phone: data.phone?.trim() || null,
            bio: data.bio?.trim() || null,
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
          userType: 'user',
        }),
      })

      toast({
        title: "Success",
        description: "Profile created successfully!",
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
    // Prevent any automatic form submission
  }

  const handleCompleteClick = async () => {
    // Explicitly trigger form submission only when Complete button is clicked
    const form = document.querySelector('form') as HTMLFormElement
    if (form && currentStep === STEPS.length - 1) {
      const isValid = await trigger(STEPS[currentStep].fields as Array<keyof IndividualProfileFormData>)
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
                  <User className="w-12 h-12 text-gray-400" />
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
            <p className="mt-3 text-sm text-gray-500 font-medium">Upload Profile Picture</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="name" className="text-base">Full Name <span className="text-red-500">*</span></Label>
            <Input
              id="name"
              {...register("name")}
              placeholder="e.g. Jane Doe"
              autoComplete="name"
              className="h-12 text-lg"
            />
            {errors.name && (
              <p className="text-sm text-destructive">{errors.name.message}</p>
            )}
          </div>
        </div>

        {/* Step 2: Basic Details */}
        <div className={cn("space-y-6", currentStep !== 1 && "hidden")}>
          <div className="space-y-2">
            <Label htmlFor="age" className="text-base">Age</Label>
            <Input
              id="age"
              type="number"
              min="1"
              max="150"
              {...register("age", { 
                valueAsNumber: true,
              })}
              placeholder="e.g. 25"
              className="h-12"
            />
            {errors.age && (
              <p className="text-sm text-destructive">{errors.age.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="location" className="text-base">Location</Label>
            <Input
              id="location"
              {...register("location")}
              placeholder="e.g. New York, USA"
              className="h-12"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="phone" className="text-base">Phone Number</Label>
            <Input
              id="phone"
              {...register("phone")}
              placeholder="e.g. +1 (555) 000-0000"
              className="h-12"
            />
          </div>
        </div>

        {/* Step 3: About */}
        <div className={cn("space-y-6", currentStep !== 2 && "hidden")}>
          <div className="space-y-2">
            <Label htmlFor="bio" className="text-base">Bio</Label>
            <Textarea
              id="bio"
              {...register("bio")}
              placeholder="Tell us a little bit about yourself..."
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
