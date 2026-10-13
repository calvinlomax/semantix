import {create} from 'zustand'
import {persist} from 'zustand/middleware'
import {api, ApiError} from '../lib/api'
import {addResult, utcDay} from '../lib/stats'
import type {CompletedGame, Difficulty, Game, ModelStatus, Mode} from '../lib/types'
export type Settings = {reducedMotion: boolean; showLabels: boolean; previewDaily: boolean; previewFree: boolean; followPlayer: boolean}
export const defaultSettings: Settings = {reducedMotion: false, showLabels: true, previewDaily: false, previewFree: true, followPlayer: false}
interface GameState {
  game: Game | null; model: ModelStatus | null; mode: Mode; difficulty: Difficulty; busy: boolean; error: string | null; setupError: string | null
  hint: string | null; hintMessage: string | null; results: CompletedGame[]; settings: Settings; activeGames: Partial<Record<Mode, string>>; savedDaily: Game | null; tutorialSeen: boolean
  initialize: () => Promise<void>; start: (mode?: Mode, difficulty?: Difficulty, retry?: boolean) => Promise<void>
  submit: (word: string, power?: boolean) => Promise<boolean>; getHint: () => Promise<void>; dismissError: () => void; updateSettings: (settings: Partial<Settings>) => void
  completeTutorial: () => void
}
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.'
export const useGameStore = create<GameState>()(persist((set, get) => ({
  game: null, model: null, mode: 'daily', difficulty: 'medium', busy: false, error: null, setupError: null, hint: null, hintMessage: null,
  results: [], settings: defaultSettings, activeGames: {}, savedDaily: null, tutorialSeen: false,
  initialize: async () => {
    if (get().busy) return
    set({busy: true, error: null, setupError: null})
    try {
      const model = await api.status(); set({model})
      const {mode, activeGames, savedDaily} = get()
      let game: Game | undefined
      if (activeGames[mode]) {
        try {const restored = await api.history(activeGames[mode]!); if (mode !== 'daily' || restored.daily_date === utcDay()) game = restored} catch { /* Expired sessions can be safely replaced. */ }
      }
      if (!game && mode === 'daily' && savedDaily?.daily_date === utcDay() && savedDaily.status !== 'playing') game = savedDaily
      if (!game) game = mode === 'daily' ? await api.daily() : await api.newGame(get().difficulty)
      set(state => ({game, model: {...model, status: 'ready'}, activeGames: {...state.activeGames, [mode]: game.id}, results: addResult(state.results, game)}))
    } catch (error) { set({setupError: message(error)}) }
    finally {set({busy: false})}
  },
  start: async (mode = get().mode, difficulty = get().difficulty, retry = false) => {
    if (get().busy) return
    const previous = get().game
    set({busy: true, error: null, setupError: null, hint: null, hintMessage: null, mode, difficulty})
    try {
      let game: Game | undefined
      if (mode !== previous?.mode && get().activeGames[mode]) {
        try {const restored = await api.history(get().activeGames[mode]!); if (mode !== 'daily' || restored.daily_date === utcDay()) game = restored} catch { /* Continue with a new session. */ }
      }
      if (mode === 'daily' && !retry && get().savedDaily?.daily_date === utcDay()) game = get().savedDaily!
      if (!game) {
        try {game = mode === 'daily' && !retry ? await api.daily() : await api.newGame(difficulty, retry ? previous?.id : undefined)}
        catch (error) {
          if (retry && mode === 'daily' && previous?.daily_date === utcDay() && error instanceof ApiError && error.code === 'game_expired') game = await api.daily()
          else throw error
        }
      }
      set(state => ({game, activeGames: {...state.activeGames, [mode]: game.id}, results: addResult(state.results, game)}))
    } catch (error) {set({error: previous ? message(error) : null, setupError: previous ? null : message(error), mode: previous?.mode ?? mode})}
    finally {set({busy: false})}
  },
  submit: async (word, power = false) => {
    const game = get().game; if (!game || game.status !== 'playing' || get().busy) return false
    set({busy: true, error: null, hint: null, hintMessage: null})
    try {
      const updated = await api.move(game, word, power)
      set(state => ({game: updated, results: addResult(state.results, updated), savedDaily: updated.mode === 'daily' && updated.status !== 'playing' ? updated : state.savedDaily}))
      return true
    } catch (error) {
      set({error: message(error)})
      if (error instanceof ApiError && ['stale_game', 'request_conflict', 'game_complete'].includes(error.code)) {
        try {const updated = await api.history(game.id); set(state => ({game: updated, results: addResult(state.results, updated)}))} catch { /* Keep the explanatory error visible. */ }
      }
      return false
    } finally {set({busy: false})}
  },
  getHint: async () => {
    const game = get().game; if (!game || get().busy || !game.hints_remaining) return
    set({busy: true, error: null})
    try {const result = await api.hint(game.id); set({hint: result.word, hintMessage: result.normalization_message, game: {...game, hints_remaining: result.remaining}})}
    catch (error) {set({error: message(error)})} finally {set({busy: false})}
  },
  dismissError: () => set({error: null}),
  updateSettings: settings => set(state => ({settings: {...state.settings, ...settings}})),
  completeTutorial: () => set({tutorialSeen: true}),
}), {
  name: 'semantix-v1',
  partialize: state => ({mode: state.mode, difficulty: state.difficulty, results: state.results, settings: state.settings, activeGames: state.activeGames, savedDaily: state.savedDaily, tutorialSeen: state.tutorialSeen}),
  merge: (persisted, current) => {
    const saved = persisted as Partial<GameState> | undefined
    return {...current, ...saved, settings: {...defaultSettings, ...saved?.settings}}
  },
}))
