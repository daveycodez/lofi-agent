/**
 * How to set up Apple's on-device model, shown when the app can't reach it.
 * The model runs through apfel (https://apfel.franzai.com), a small Mac app
 * that serves it on localhost; the browser talks to it directly.
 */
import { Button, Link, Modal } from "@heroui/react"
import { useSyncExternalStore } from "react"

import { Markdown } from "#/components/chat/markdown"

/** iPhone or iPad (iPadOS reports itself as a Mac, but has touch). */
const isIOS = () =>
  /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (/Mac/.test(navigator.userAgent) && navigator.maxTouchPoints > 1)

const noopSubscribe = () => () => undefined

/** The command that starts apfel for this site. */
function serveCommand() {
  const { origin, hostname } = window.location
  // apfel allows localhost pages by default; anything else is added by name.
  const isLocal = ["localhost", "127.0.0.1", "[::1]"].includes(hostname)
  return isLocal
    ? "apfel --serve --cors"
    : `apfel --serve --cors --allowed-origins ${origin}`
}

/** A shell command in a code block, with the copy button code blocks have. */
function Command({ children }: { children: string }) {
  return (
    <div className="markdown prose prose-sm max-w-none">
      <Markdown>{`\`\`\`bash\n${children}\n\`\`\``}</Markdown>
    </div>
  )
}

export function AppleSetupModal({
  isOpen,
  onOpenChange,
  onRetry,
  isChecking
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** Checks for the model again. */
  onRetry: () => void
  isChecking: boolean
}) {
  const ios = useSyncExternalStore(noopSubscribe, isIOS, () => false)
  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange}>
      <Modal.Container size="md" scroll="inside">
        <Modal.Dialog>
          <Modal.CloseTrigger />
          <Modal.Header>
            <Modal.Heading>Set Up Apple Intelligence</Modal.Heading>
          </Modal.Header>
          {ios ? (
            <Modal.Body className="text-sm text-muted">
              Apple's on-device model runs through apfel on a Mac, so it isn't
              available in Safari on iPhone or iPad. Open this site on your Mac
              to use it, or pick another model.
            </Modal.Body>
          ) : (
            <Modal.Body className="flex flex-col gap-4 text-sm">
              <p className="text-muted">
                Apple's model runs on your Mac through{" "}
                <Link
                  href="https://apfel.franzai.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  apfel
                </Link>
                , a free app that makes it available to this page. You need a
                Mac with Apple silicon on macOS 26 or later, with Apple
                Intelligence turned on in System Settings.
              </p>
              <div className="flex flex-col gap-2">
                <p className="font-medium">1. Install apfel with Homebrew</p>
                <Command>brew install apfel</Command>
              </div>
              <div className="flex flex-col gap-2">
                <p className="font-medium">
                  2. Start it, and leave the Terminal window open
                </p>
                <Command>{serveCommand()}</Command>
              </div>
              <p className="text-muted">
                3. Click Try Again. If your browser asks to access devices on
                your local network, allow it.
              </p>
            </Modal.Body>
          )}
          <Modal.Footer>
            <Button slot="close" variant="tertiary">
              {ios ? "Close" : "Cancel"}
            </Button>
            {!ios && (
              <Button isPending={isChecking} onPress={onRetry}>
                Try Again
              </Button>
            )}
          </Modal.Footer>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  )
}
