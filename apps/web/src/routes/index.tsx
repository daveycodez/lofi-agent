import { Button, Card } from "@heroui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"
import { ThemeSwitcher } from "#/components/theme-switcher"

export const Route = createFileRoute("/")({ component: Home })

function Home() {
  const [count, setCount] = useState(0)

  return (
    <main className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <Card.Header>
          <Card.Title>lofi-agent</Card.Title>
          <Card.Description>
            HeroUI is installed. This card and button are HeroUI components.
          </Card.Description>
        </Card.Header>
        <Card.Content>
          <ThemeSwitcher />
        </Card.Content>
        <Card.Footer className="flex gap-2">
          <Button onPress={() => setCount((c) => c + 1)}>
            Pressed {count} {count === 1 ? "time" : "times"}
          </Button>
          <Button variant="secondary" onPress={() => setCount(0)}>
            Reset
          </Button>
        </Card.Footer>
      </Card>
    </main>
  )
}
