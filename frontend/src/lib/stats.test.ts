import {describe, expect, it} from 'vitest'
import {addResult, calculateStats, shareText, utcDay} from './stats'
import type {CompletedGame, Game} from './types'

const result = (day: string, status: 'won' | 'lost' = 'won'): CompletedGame => ({id: day, mode: 'daily', daily_date: day, status, moves: 3, similarity: 0.8, rating: 'Brilliant', completed_at: `${day}T12:00:00Z`})
const game = (overrides = {}): Game => ({id: 'one', mode: 'daily', daily_date: '2026-10-13', status: 'won', moves_used: 3, move_limit: 6, similarity: .8, rating: 'Brilliant', history: [{delta: .2}, {delta: -.1}], ...overrides} as Game)

describe('persistent game statistics', () => {
  it('records only completed outcomes and deduplicates daily practices', () => {
    expect(addResult([], game({status: 'playing'}))).toEqual([])
    const first = addResult([], game({status: 'lost'}))
    expect(addResult(first, game({id: 'retry', status: 'won'}))).toEqual(first)
    expect(addResult(first, game({id: 'free', mode: 'free', daily_date: null}))).toHaveLength(2)
  })
  it('deduplicates free games by ID on refresh', () => {
    const free = game({mode: 'free', daily_date: null})
    const results = addResult([], free)
    expect(addResult(results, free)).toHaveLength(1)
  })
  it('computes wins, best result and average winning moves', () => {
    const stats = calculateStats([result('2026-10-11'), result('2026-10-12', 'lost'), {...result('2026-10-13'), moves: 5}], new Date('2026-10-13T20:00:00Z'))
    expect(stats).toMatchObject({played: 3, won: 2, winPercentage: 67, averageMoves: '4.0', best: 3, streak: 1, longestStreak: 1})
  })
  it('uses UTC dates and keeps yesterday’s streak alive until today ends', () => {
    expect(utcDay(new Date('2026-10-12T23:30:00-04:00'))).toBe('2026-10-13')
    const results = [result('2026-10-11'), result('2026-10-12')]
    expect(calculateStats(results, new Date('2026-10-13T23:59:59Z')).streak).toBe(2)
    expect(calculateStats(results, new Date('2026-10-14T00:00:00Z')).streak).toBe(0)
    expect(calculateStats(results).longestStreak).toBe(2)
  })
  it('shares outcomes without puzzle or submitted words', () => {
    const summary = shareText(game({start_word: 'secretstart', target_word: 'secrettarget', history: [{word: 'secretmove', delta: .2}]}))
    expect(summary).toContain('3/6')
    expect(summary).toContain('2026-10-13')
    expect(summary).not.toContain('secret')
  })
})
