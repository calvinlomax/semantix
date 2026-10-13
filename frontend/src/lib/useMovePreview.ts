import {useEffect, useRef, useState} from 'react'
import {api} from './api'
import type {Game, MovePreview} from './types'

interface PreviewState {key: string; preview: MovePreview | null; error: string | null; pending: boolean}
const empty = {key: '', preview: null, error: null, pending: false}

/** Both request cancellation and a generation check protect against late replies. */
export function useMovePreview(game: Game | null, word: string, power: boolean, enabled: boolean) {
  const [state, setState] = useState<PreviewState>(empty)
  const generation = useRef(0)
  const key = enabled && game?.status === 'playing' && word.trim()
    ? JSON.stringify([game.id, game.moves_used, word, power]) : ''
  useEffect(() => {
    const requestGeneration = ++generation.current
    const controller = new AbortController()
    if (!key || !game) {setState(empty); return () => controller.abort()}
    setState({key, preview: null, error: null, pending: true})
    const timer = window.setTimeout(async () => {
      try {
        const preview = await api.preview(game, word, power, controller.signal)
        if (!controller.signal.aborted && generation.current === requestGeneration) setState({key, preview, error: null, pending: false})
      } catch (error) {
        if (!controller.signal.aborted && generation.current === requestGeneration) setState({key, preview: null, error: error instanceof Error ? error.message : 'Preview unavailable. You can still submit your word.', pending: false})
      }
    }, 350)
    return () => {window.clearTimeout(timer); controller.abort()}
    // The key includes every value that can affect movement; hint changes do not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state.key === key ? state : empty
}
