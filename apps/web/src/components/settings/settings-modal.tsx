/** The Settings modal: web search, and a custom system prompt for every model. */
import {
  Button,
  Description,
  Label,
  Modal,
  Switch,
  TextArea,
  TextField
} from "@heroui/react"
import { useState } from "react"

import { systemPromptStore, webSearchEnabledStore } from "#/lib/settings"

export function SettingsModal({
  isOpen,
  onOpenChange
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
}) {
  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange}>
      <Modal.Container size="lg" scroll="inside">
        <Modal.Dialog>
          {({ close }) => <SettingsForm onDone={close} />}
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
function SettingsForm({ onDone }: { onDone: () => void }) {
  const [webSearch, setWebSearch] = useState(webSearchEnabledStore.get)
  const [systemPrompt, setSystemPrompt] = useState(systemPromptStore.get)
  return (
    <>
      <Modal.CloseTrigger />
      <Modal.Header>
        <Modal.Heading>Settings</Modal.Heading>
      </Modal.Header>
      <Modal.Body className="flex flex-col gap-6">
        <Switch isSelected={webSearch} onChange={setWebSearch}>
          <Switch.Content className="w-full justify-between">
            <div className="flex flex-col gap-0.5">
              <Label>Web Search</Label>
              <Description>
                Lets models search the web when a question needs it. Tries Exa,
                then Parallel, then Firecrawl.
              </Description>
            </div>
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Content>
        </Switch>
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
            Sent to every model at the start of each chat. Leave it empty to use
            none. Kept in this browser only.
          </Description>
        </TextField>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="tertiary" onPress={onDone}>
          Cancel
        </Button>
        <Button
          onPress={() => {
            webSearchEnabledStore.set(webSearch)
            systemPromptStore.set(systemPrompt)
            onDone()
          }}
        >
          Save
        </Button>
      </Modal.Footer>
    </>
  )
}
