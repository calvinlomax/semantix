import type {CompletedGame, Game} from './types'
export function utcDay(now = new Date()) {return now.toISOString().slice(0, 10)}
export function gameKey(game: Pick<Game, 'id' | 'mode' | 'daily_date'>) {return game.mode === 'daily' ? `daily:${game.daily_date}` : game.id}
export function addResult(results: CompletedGame[], game: Game, now = new Date()): CompletedGame[] {
  if (game.status === 'playing' || results.some(result => gameKey(result) === gameKey(game))) return results
  return [...results, {id: game.id, daily_date: game.daily_date, mode: game.mode, status: game.status, moves: game.moves_used, similarity: game.similarity, rating: game.rating, completed_at: now.toISOString()}].slice(-3000)
}
export function calculateStats(results: CompletedGame[], now = new Date()) {
  const wins = results.filter(game => game.status === 'won')
  const days = [...new Set(wins.filter(game => game.mode === 'daily' && game.daily_date).map(game => game.daily_date!))].sort()
  let longestStreak = 0, run = 0, previous = 0
  for (const day of days) {const value = Date.parse(day); run = previous && value - previous === 86400000 ? run + 1 : 1; longestStreak = Math.max(longestStreak, run); previous = value}
  const today = Date.parse(utcDay(now)); let check = days.includes(utcDay(now)) ? today : today - 86400000; let streak = 0
  const wonDays = new Set(days)
  while (wonDays.has(new Date(check).toISOString().slice(0, 10))) {streak++; check -= 86400000}
  return {played: results.length, won: wins.length, winPercentage: results.length ? Math.round(wins.length / results.length * 100) : 0,
    averageMoves: wins.length ? (wins.reduce((sum, game) => sum + game.moves, 0) / wins.length).toFixed(1) : '—',
    best: wins.length ? Math.min(...wins.map(game => game.moves)) : null, streak, longestStreak}
}
export function shareText(game: Game) {
  const dots = game.history.map(move => move.delta > 0.005 ? '🟪' : move.delta < -0.005 ? '⬛' : '⬜').join('')
  return `SEMANTIX ${game.mode === 'daily' ? game.daily_date : 'Free Play'}\n${game.status === 'won' ? `${game.moves_used}/${game.move_limit} · Target reached` : `×/${game.move_limit} · A new route awaits`}\n${dots}\nNavigate the meaning of words.`
}
