/**
 * The sessions writing a reply right now, for the sidebar to shimmer their
 * titles. Only the open session runs, so this holds at most one, but it's a
 * set so nothing assumes that.
 */
import { createStore } from "#/lib/store"

const generating = createStore<ReadonlySet<string>>(new Set())

export function setGenerating(threadId: string, isGenerating: boolean) {
  if (generating.get().has(threadId) === isGenerating) return
  generating.set((ids) => {
    const next = new Set(ids)
    if (isGenerating) next.add(threadId)
    else next.delete(threadId)
    return next
  })
}

export const useIsGenerating = (threadId: string) =>
  generating.use().has(threadId)
