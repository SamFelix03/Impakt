// Cache for geocoded locations to avoid repeated API calls
const geocodeCache = new Map<string, [number, number]>()

// Default coordinates (center of world map)
const DEFAULT_COORDS: [number, number] = [0, 20]

export async function geocodeLocation(location: string): Promise<[number, number]> {
  // Check cache first
  if (geocodeCache.has(location)) {
    return geocodeCache.get(location)!
  }

  try {
    const response = await fetch(`/api/geocode?address=${encodeURIComponent(location)}`)
    const result = await response.json()

    if (result.success && result.coordinates) {
      const coords: [number, number] = result.coordinates
      // Cache the result
      geocodeCache.set(location, coords)
      return coords
    } else {
      console.warn(`Geocoding failed for location: ${location}, using default`)
      return DEFAULT_COORDS
    }
  } catch (error) {
    console.error('Error geocoding location:', error)
    return DEFAULT_COORDS
  }
}
