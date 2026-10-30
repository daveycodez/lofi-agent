import { useEffect, useRef, useState, useSyncExternalStore } from "react"

/** The bits of the Web Speech API we use (not in every TS DOM lib yet). */
type Recognition = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult:
    | ((event: {
        results: ArrayLike<ArrayLike<{ transcript: string }>>
      }) => void)
    | null
  onend: (() => void) | null
  onerror: ((event: { error: string }) => void) | null
  start: () => void
  stop: () => void
  abort: () => void
}

type RecognitionConstructor = new () => Recognition

function recognitionConstructor(): RecognitionConstructor | undefined {
  if (typeof window === "undefined") return undefined
  const w = window as unknown as {
    SpeechRecognition?: RecognitionConstructor
    webkitSpeechRecognition?: RecognitionConstructor
  }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition
}

const noopSubscribe = () => () => undefined

/**
 * Speech-to-text with the browser's Web Speech API. `start` calls `onText`
 * with the full transcript so far (interim results included) as you talk.
 */
export function useDictation() {
  const isSupported = useSyncExternalStore(
    noopSubscribe,
    () => !!recognitionConstructor(),
    () => false
  )
  const [isListening, setIsListening] = useState(false)
  const recognition = useRef<Recognition | null>(null)

  const cancel = () => {
    const rec = recognition.current
    if (!rec) return
    // Drop any late results so they can't refill a cleared composer.
    rec.onresult = null
    rec.abort()
  }

  const start = (onText: (text: string) => void) => {
    const Ctor = recognitionConstructor()
    if (!Ctor || recognition.current) return
    const rec = new Ctor()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = navigator.language
    rec.onresult = (event) => {
      let text = ""
      for (const result of Array.from(event.results)) {
        text += result[0]?.transcript ?? ""
      }
      onText(text)
    }
    rec.onend = () => {
      recognition.current = null
      setIsListening(false)
    }
    rec.onerror = () => rec.onend?.()
    recognition.current = rec
    setIsListening(true)
    rec.start()
  }

  const stop = () => recognition.current?.stop()

  useEffect(() => () => recognition.current?.abort(), [])

  return { isSupported, isListening, start, stop, cancel }
}
