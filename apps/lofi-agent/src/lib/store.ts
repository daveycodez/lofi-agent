import { useSyncExternalStore } from "react"

/**
 * Tiny in-memory store, for state that lasts only while the page is open
 * (connection and loading status). Saved state lives in TanStack DB.
 */
export function createStore<T>(initial: T) {
  let value = initial
  const listeners = new Set<() => void>()

  const get = () => value

  const set = (next: T | ((prev: T) => T)) => {
    value = typeof next === "function" ? (next as (prev: T) => T)(value) : next
    listeners.forEach((l) => l())
  }

  const subscribe = (listener: () => void) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }

  const use = () => useSyncExternalStore(subscribe, get, () => initial)

  return { get, set, subscribe, use }
}
