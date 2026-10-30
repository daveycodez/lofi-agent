import { Button } from "@heroui/react"
import type { ReactNode } from "react"

export function Suggestions({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3 flex flex-wrap justify-center gap-2">{children}</div>
  )
}

/** A prompt to start with, sent when pressed. */
export function Suggestion({
  children,
  onPress
}: {
  children: ReactNode
  onPress: () => void
}) {
  return (
    <Button size="sm" variant="outline" onPress={onPress}>
      {children}
    </Button>
  )
}
