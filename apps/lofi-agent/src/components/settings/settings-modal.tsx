/**
 * The Settings modal: a custom system prompt for every model, and deleting
 * every session.
 */
import TrashBin from "@gravity-ui/icons/TrashBin"
import {
  AlertDialog,
  Button,
  Description,
  Label,
  Modal,
  TextArea,
  TextField
} from "@heroui/react"
import { useLiveQuery, useOptionalDbClient } from "@tanstack/react-db"
import { useState } from "react"

import { threadsCollection } from "#/lib/db"
import { systemPromptSetting } from "#/lib/settings"
import { deleteAllThreads } from "#/lib/threads"

export function SettingsModal({
  isOpen,
  onOpenChange,
  onAllDeleted
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** Called once every session is deleted (the app opens a new one). */
  onAllDeleted: () => void
}) {
  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange}>
      <Modal.Container size="lg" scroll="inside">
        <Modal.Dialog>
          {({ close }) => (
            <SettingsForm
              onDone={close}
              onAllDeleted={() => {
                close()
                onAllDeleted()
              }}
            />
          )}
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  )
}

/**
 * Edits a copy of the settings: Save keeps the changes, Cancel (or closing
 * the modal) drops them. Mounted each time the modal opens, so it starts
 * from the saved values.
 */
function SettingsForm({
  onDone,
  onAllDeleted
}: {
  onDone: () => void
  onAllDeleted: () => void
}) {
  const [systemPrompt, setSystemPrompt] = useState(systemPromptSetting.get)
  return (
    <>
      <Modal.CloseTrigger />
      <Modal.Header>
        <Modal.Heading>Settings</Modal.Heading>
      </Modal.Header>
      <Modal.Body className="flex flex-col gap-6">
        <TextField
          variant="secondary"
          value={systemPrompt}
          onChange={setSystemPrompt}
        >
          <Label>System prompt</Label>
          <TextArea
            rows={6}
            placeholder="For example: Answer briefly, in plain English."
            className="resize-y"
          />
          <Description>
            Sent to every model at the start of each session, before
            today&apos;s date. Kept in this browser only.
          </Description>
        </TextField>
        <DeleteAllSessions onDeleted={onAllDeleted} />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="tertiary" onPress={onDone}>
          Cancel
        </Button>
        <Button
          onPress={() => {
            systemPromptSetting.set(systemPrompt)
            onDone()
          }}
        >
          Save
        </Button>
      </Modal.Footer>
    </>
  )
}

/**
 * Deletes every session, archived ones included, once confirmed. Takes
 * effect right away, apart from the form's Save.
 */
function DeleteAllSessions({ onDeleted }: { onDeleted: () => void }) {
  const db = useOptionalDbClient()
  const { data: threads } = useLiveQuery({
    query: (q) => (db ? q.from({ thread: threadsCollection }) : null)
  })
  const count = threads?.length ?? 0
  const [isConfirming, setIsConfirming] = useState(false)
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex min-w-0 flex-col gap-0.5">
        <p className="text-sm font-medium">Delete all sessions</p>
        <p className="text-xs text-muted">
          Permanently deletes every session, archived ones included.
        </p>
      </div>
      <Button
        size="sm"
        variant="danger-soft"
        isDisabled={count === 0}
        onPress={() => setIsConfirming(true)}
        className="shrink-0"
      >
        Delete all
      </Button>

      <AlertDialog.Backdrop
        isOpen={isConfirming}
        onOpenChange={setIsConfirming}
      >
        <AlertDialog.Container>
          <AlertDialog.Dialog className="sm:max-w-[400px]">
            <AlertDialog.CloseTrigger />
            <AlertDialog.Header>
              <AlertDialog.Icon status="danger">
                <TrashBin className="size-5" />
              </AlertDialog.Icon>
              <AlertDialog.Heading>Delete all sessions?</AlertDialog.Heading>
            </AlertDialog.Header>
            <AlertDialog.Body>
              <p>
                This permanently deletes{" "}
                {count === 1 ? "your session" : `all ${count} sessions`},
                including archived ones. This can&apos;t be undone.
              </p>
            </AlertDialog.Body>
            <AlertDialog.Footer>
              <Button slot="close" variant="tertiary">
                Cancel
              </Button>
              <Button
                slot="close"
                variant="danger"
                onPress={() => {
                  deleteAllThreads()
                  onDeleted()
                }}
              >
                Delete All
              </Button>
            </AlertDialog.Footer>
          </AlertDialog.Dialog>
        </AlertDialog.Container>
      </AlertDialog.Backdrop>
    </div>
  )
}
