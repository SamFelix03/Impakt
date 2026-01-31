"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/auth"
import { Loader2 } from "lucide-react"

export default function HomePage() {
  const router = useRouter()
  const { ready, authenticated, profileComplete, loading, login } = useAuth()
  const [svgContent, setSvgContent] = useState<string>("")
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    // Always use light theme
    document.documentElement.classList.remove("dark")
  }, [])

  // Handle redirects after authentication - only redirect authenticated users away from landing page
  useEffect(() => {
    if (!ready || loading) return
    
    if (authenticated) {
      // If user is authenticated but profile is not complete, redirect to signup
      if (!profileComplete) {
        router.replace("/signup")
      } else {
        // If profile is complete, redirect to home
        router.replace("/home")
      }
    }
    // If not authenticated, stay on landing page (no redirect)
  }, [ready, authenticated, profileComplete, loading, router])

  useEffect(() => {
    const loadSVG = async () => {
      try {
        const response = await fetch("/map.svg")
        const svgText = await response.text()
        setSvgContent(svgText)
      } catch (error) {
        console.error("Failed to load SVG:", error)
      }
    }

    if (mounted) {
      loadSVG()
    }
  }, [mounted])

  useEffect(() => {
    if (svgContent) {
      const timer = setTimeout(() => {
        const rects = document.querySelectorAll("#map-svg rect")
        console.log(`[v0] Found ${rects.length} rect elements`)

        rects.forEach((rect, index) => {
          const duration = Math.random() * 1.5 + 0.5 // 0.5-2 seconds (faster)
          const delay = Math.random() * 1 // 0-1 seconds

          rect.setAttribute(
            "style",
            `
            animation: glimmer ${duration}s ease-in-out ${delay}s infinite alternate;
          `,
          )
        })

        const style = document.createElement("style")
        style.textContent = `
          @keyframes glimmer {
            0% { opacity: 1; }
            100% { opacity: 0.1; }
          }
        `
        document.head.appendChild(style)
      }, 100)

      return () => clearTimeout(timer)
    }
  }, [svgContent])

  // Show loading state while checking auth
  if (!ready || loading || !mounted) {
    return (
      <div
        className="min-h-screen w-full overflow-x-hidden flex items-center justify-center p-4 relative"
        style={{ backgroundColor: "#ffffff" }}
      >
        <div className="flex flex-col items-center justify-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  // If authenticated but profile not complete, show loading (will redirect)
  if (authenticated && !profileComplete) {
    return (
      <div
        className="min-h-screen w-full overflow-x-hidden flex items-center justify-center p-4 relative"
        style={{ backgroundColor: "#ffffff" }}
      >
        <div className="flex flex-col items-center justify-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Setting up your account...</p>
        </div>
      </div>
    )
  }

  return (
    <div
      className="min-h-screen w-full overflow-x-hidden flex items-center justify-center p-4 relative"
      style={{ backgroundColor: "#ffffff" }}
    >
      {/* Background Map SVG */}
      <div className="absolute inset-0 flex justify-center items-center w-full pointer-events-none">
        {svgContent ? (
          <div
            id="map-svg"
            className="overflow-hidden flex items-center justify-center opacity-45"
            dangerouslySetInnerHTML={{ __html: svgContent }}
            style={{ width: "1200px", height: "1200px", maxWidth: "1200px", maxHeight: "1200px" }}
          />
        ) : (
          <div
            className="bg-gray-200 animate-pulse rounded-lg"
            style={{ width: "1200px", height: "1200px", maxWidth: "1200px", maxHeight: "1200px" }}
          />
        )}
      </div>

      {/* Foreground Content */}
      <div className="relative z-10 flex flex-col items-center justify-center gap-6 text-center">
        <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold text-green-800 fade-in-up">
          impakt
        </h1>
        
        <p className="text-xl md:text-2xl lg:text-3xl text-muted-foreground fade-in-up-delay-1">
          Help, and enable others that help!
        </p>
        
        <div className="mt-4 fade-in-up-delay-2">
          <Button 
            size="lg" 
            className="text-lg px-8 py-6"
            onClick={() => {
              if (authenticated && profileComplete) {
                // User is logged in and has profile, can stay here or go to a main page
                // For now, just show a message or you can create a dashboard later
                console.log("User is logged in with profile")
              } else {
                login()
              }
            }}
          >
            {authenticated && profileComplete ? "Welcome Back!" : "Get Started"}
          </Button>
        </div>
      </div>
    </div>
  )
}
