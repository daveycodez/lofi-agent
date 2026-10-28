import { Button } from "@heroui/react"
import { useTheme } from "next-themes"
import { useSyncExternalStore } from "react"

const THEMES = ["system", "light", "dark"] as const

// Nothing to subscribe to: the value only changes between server and client render.
const noopSubscribe = () => () => undefined

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme()
  // The saved theme is only known on the client: false during SSR/hydration, true after.
  const mounted = useSyncExternalStore(
    noopSubscribe,
    () => true,
    () => false
  )

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
