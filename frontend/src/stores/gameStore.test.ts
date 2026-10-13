import {beforeEach, describe, expect, it, vi} from 'vitest'
import {api} from '../lib/api'
import type {Game} from '../lib/types'

vi.mock('../lib/api', async importOriginal => ({...await importOriginal<typeof import('../lib/api')>(), api: {move: vi.fn(), hint: vi.fn()}}))
const saved = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (key: string) => saved.get(key) ?? null,
  setItem: (key: string, value: string) => saved.set(key, value),
  removeItem: (key: string) => saved.delete(key),
})
vi.stubGlobal('window', {localStorage: globalThis.localStorage})
const {useGameStore, defaultSettings} = await import('./gameStore')
const game = {id: 'test', mode: 'free', status: 'playing', moves_used: 0, history: [], hints_remaining: 2} as unknown as Game

beforeEach(() => {
  vi.clearAllMocks(); saved.clear()
  useGameStore.setState({game, busy: false, error: null, hint: null, hintMessage: null, results: [], savedDaily: null, settings: {...defaultSettings}, tutorialSeen: false})
})

describe('game controls and persisted preferences', () => {
  it('keeps Daily previews off and Free Play previews on, independently', () => {
    expect(useGameStore.getState().settings).toMatchObject({previewDaily: false, previewFree: true})
    useGameStore.getState().updateSettings({previewDaily: true})
    useGameStore.getState().updateSettings({previewFree: false})
    expect(useGameStore.getState().settings).toMatchObject({previewDaily: true, previewFree: false})
  })
  it('adds new preference defaults when restoring old saved settings', async () => {
    saved.set('semantix-v1', JSON.stringify({state: {settings: {reducedMotion: true, showLabels: false}}, version: 0}))
    await useGameStore.persist.rehydrate()
    expect(useGameStore.getState().settings).toEqual({...defaultSettings, reducedMotion: true, showLabels: false})
  })
  it('prevents duplicate submissions while preserving server word resolution', async () => {
    let finish!: (value: Game) => void
    vi.mocked(api.move).mockReturnValue(new Promise(resolve => {finish = resolve}))
    const first = useGameStore.getState().submit('Prepared')
    expect(useGameStore.getState().busy).toBe(true)
    expect(await useGameStore.getState().submit('Prepared')).toBe(false)
    expect(api.move).toHaveBeenCalledTimes(1)
    expect(api.move).toHaveBeenCalledWith(game, 'Prepared', false)
    const updated = {...game, moves_used: 1, history: [{number: 1, word: 'Prepared', resolved_word: 'prepare', resolution: 'lemma', normalization_message: '"Prepared" accepted using "prepare"'}]} as Game
    finish(updated)
    expect(await first).toBe(true)
    expect(useGameStore.getState().game?.history[0].word).toBe('Prepared')
    expect(useGameStore.getState().game?.history[0].resolved_word).toBe('prepare')
    expect(useGameStore.getState().busy).toBe(false)
  })
  it('keeps hint resolution and persists tutorial completion', async () => {
    vi.mocked(api.hint).mockResolvedValue({word: 'prepared', resolved_word: 'prepare', resolution: 'lemma', normalization_message: '"prepared" accepted using "prepare"', remaining: 1})
    await useGameStore.getState().getHint()
    expect(useGameStore.getState()).toMatchObject({hint: 'prepared', hintMessage: '"prepared" accepted using "prepare"', game: {hints_remaining: 1}})
    useGameStore.getState().completeTutorial()
    expect(JSON.parse(saved.get('semantix-v1')!).state.tutorialSeen).toBe(true)
  })
})
