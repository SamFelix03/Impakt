'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState, useRef } from 'react'
import { User, LogOut } from 'lucide-react'
import { useAuth } from '@/frontend/lib/auth'
import { Button } from '@/frontend/components/ui/button'

export default function Navbar() {
  const pathname = usePathname()
  const router = useRouter()
  const { authenticated, userProfile, orgProfile, logout, dbUser } = useAuth()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Close dropdown when clicking outside
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleLogout = async () => {
    setDropdownOpen(false)
    await logout()
    router.push('/')
    router.refresh()
  }

  // Don't show navbar on auth pages
  if (pathname === '/signup' || pathname === '/') {
    return null
  }

  const profile = userProfile || orgProfile
  const displayName = profile?.name || dbUser?.email?.split('@')[0] || 'User'
  const profilePic = profile?.profile_pic_url

  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      {/* Blur backdrop */}
      <div className="absolute inset-0 bg-white/80 backdrop-blur-md"></div>
      
      {/* Navbar content */}
      <div className="relative max-w-6xl mx-auto px-4 py-3">
        <div className="bg-white/90 rounded-full shadow-lg border border-gray-200 px-6 py-3 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          {/* Left Side - Logo */}
          <div className="flex items-center">
            <Link 
              href="/home" 
              className="flex items-center gap-2 hover:opacity-80 transition-opacity"
            >
              <span className="text-2xl font-bold text-green-800">impakt</span>
            </Link>
          </div>

          {/* Center Navigation - Always centered */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <div className="flex items-center gap-1">
              <Link
                href="/home"
                className={`px-5 py-2.5 rounded-full text-base font-medium transition-colors ${
                  pathname === '/home'
                    ? 'bg-green-800 text-white'
                    : 'text-gray-700 hover:bg-green-50 hover:text-green-800'
                }`}
              >
                Home
              </Link>
              <Link
                href="/dashboard"
                className={`px-5 py-2.5 rounded-full text-base font-medium transition-colors ${
                  pathname === '/dashboard'
                    ? 'bg-green-800 text-white'
                    : 'text-gray-700 hover:bg-green-50 hover:text-green-800'
                }`}
              >
                Dashboard
              </Link>
            </div>
          </div>
          
          {/* Right Side - Profile */}
          <div className="flex items-center gap-3">
            {/* Profile Dropdown */}
            {authenticated && (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className="flex items-center gap-2 px-3 py-2 rounded-full hover:bg-green-50 transition-colors"
                >
                  {profilePic ? (
                    <img
                      src={profilePic}
                      alt="Profile"
                      className="w-8 h-8 rounded-full object-cover border-2 border-green-800"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-green-800 flex items-center justify-center">
                      <User className="w-5 h-5 text-white" />
                    </div>
                  )}
                </button>

                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-56 rounded-xl shadow-lg bg-white ring-1 ring-gray-200 z-50 overflow-hidden">
                    <div className="py-1">
                      <div className="px-4 py-3 border-b border-gray-200">
                        <p className="text-sm font-medium text-gray-900">
                          {displayName}
                        </p>
                        {dbUser?.email && (
                          <p className="text-xs text-gray-500 mt-1 truncate">
                            {dbUser.email}
                          </p>
                        )}
                      </div>
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <LogOut className="w-4 h-4 mr-3" />
                        Sign Out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        </div>
      </div>
    </nav>
  )
}
