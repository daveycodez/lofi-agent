import { useSyncExternalStore } from "react"

/**
 * Tiny external store. Optionally persisted to localStorage under `key`.
 * SSR-safe: the server (and first client render) always see `initial`.
 */
export function createStore<T>(initial: T, key?: string) {
  let value = initial
  let loaded = !key
  const listeners = new Set<() => void>()

  const load = () => {
    if (loaded || typeof window === "undefined") return
    loaded = true
    try {
      const raw = key ? window.localStorage.getItem(key) : null
      if (raw) value = JSON.parse(raw) as T
    } catch {
      // Storage blocked or corrupt: keep the default.
    }
  }

  const get = () => {
    load()
    return value
  }

  const set = (next: T | ((prev: T) => T)) => {
    load()
    value = typeof next === "function" ? (next as (prev: T) => T)(value) : next
    if (key) {
      try {
        window.localStorage.setItem(key, JSON.stringify(value))
      } catch {
        // Ignore quota / private-mode failures; state still works in memory.
      }
    }
    listeners.forEach((l) => l())
  }

  const subscribe = (listener: () => void) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }

  const use = () => useSyncExternalStore(subscribe, get, () => initial)

  return { get, set, subscribe, use }
}
