import { create } from 'zustand'
import type { AgentStep, SourceRef, ImageRef } from '../types'

interface ChatState {
  isStreaming:      boolean
  streamingContent: string
  agentSteps:       AgentStep[]
  currentAgentStep: string | null
  pendingSources:   SourceRef[]
  pendingImages:    ImageRef[]
  error:            string | null

  startStream:      () => void
  appendToken:      (token: string) => void
  updateAgentStep:  (step: string, label: string, done: boolean, elapsed?: number) => void
  setPendingSources:(sources: SourceRef[]) => void
  setPendingImages: (images: ImageRef[]) => void
  finalizeStream:   () => { content: string; sources: SourceRef[]; images: ImageRef[] }
  resetStream:      () => void
  setError:         (msg: string) => void
}

const AGENT_ORDER = ['retriever', 'procedure', 'safety', 'parts', 'simplifier', 'editor']

export const useChatStore = create<ChatState>((set, get) => ({
  isStreaming:      false,
  streamingContent: '',
  agentSteps:       [],
  currentAgentStep: null,
  pendingSources:   [],
  pendingImages:    [],
  error:            null,

  startStream: () => set({
    isStreaming:      true,
    streamingContent: '',
    agentSteps:       [],
    currentAgentStep: null,
    pendingSources:   [],
    pendingImages:    [],
    error:            null,
  }),

  appendToken: (token) => set(state => ({
    streamingContent: state.streamingContent + token,
  })),

  updateAgentStep: (step, label, done, elapsed) => set(state => {
    const existingIdx = state.agentSteps.findIndex(s => s.step === step)
    const stepKey = step as AgentStep['step']
    let agentSteps: AgentStep[]

    if (existingIdx >= 0) {
      agentSteps = state.agentSteps.map((s, i) =>
        i === existingIdx ? { ...s, done, elapsed: elapsed ?? s.elapsed } : s
      )
    } else {
      // Insert in defined order
      const newStep: AgentStep = { step: stepKey, label, done, elapsed }
      agentSteps = [...state.agentSteps, newStep]
        .sort((a, b) => AGENT_ORDER.indexOf(a.step) - AGENT_ORDER.indexOf(b.step))
    }

    return {
      agentSteps,
      currentAgentStep: done ? null : step,
    }
  }),

  setPendingSources: (sources) => set({ pendingSources: sources }),
  setPendingImages:  (images)  => set({ pendingImages:  images  }),

  finalizeStream: () => {
    const { streamingContent, pendingSources, pendingImages } = get()
    set({ isStreaming: false })
    return { content: streamingContent, sources: pendingSources, images: pendingImages }
  },

  resetStream: () => set({
    isStreaming:      false,
    streamingContent: '',
    agentSteps:       [],
    currentAgentStep: null,
    pendingSources:   [],
    pendingImages:    [],
    error:            null,
  }),

  setError: (msg) => set({ error: msg, isStreaming: false }),
}))
