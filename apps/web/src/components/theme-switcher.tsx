import { Button } from "@heroui/react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

const THEMES = ["system", "light", "dark"] as const

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme()
  // The saved theme is only known on the client; render a stable state until mounted.
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  return (
    <div className="flex gap-2" role="group" aria-label="Theme">
      {THEMES.map((t) => (
        <Button
          key={t}
          size="sm"
          variant={mounted && theme === t ? "primary" : "tertiary"}
          aria-pressed={mounted && theme === t}
          onPress={() => setTheme(t)}
        >
          {t[0].toUpperCase() + t.slice(1)}
        </Button>
      ))}
    </div>
  )
}
