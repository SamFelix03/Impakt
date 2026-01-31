import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import "./globals.css"
import { Providers } from "./providers"
import { Toaster } from "@/components/ui/toaster"

export const metadata: Metadata = {
  title: "impakt",
  description: "impakt",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${GeistSans.variable} ${GeistMono.variable} antialiased`} style={{ fontFamily: "'Futura PT', sans-serif" }}>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  // Ensure light theme - remove any dark class
                  document.documentElement.classList.remove('dark');
                  document.documentElement.style.colorScheme = 'light';
                } catch (e) {}
              })();
            `,
          }}
        />
        <Providers>
          {children}
          <Suspense fallback={null}>
            <Analytics />
          </Suspense>
          <Toaster />
        </Providers>
      </body>
    </html>
  )
}
