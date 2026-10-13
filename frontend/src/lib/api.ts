import type { Difficulty, Game, Hint, ModelStatus, MovePreview } from './types'
export class ApiError extends Error { constructor(public code: string, message: string) {super(message); this.name = 'ApiError'} }
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response
  try { response = await fetch(`/api${path}`, body === undefined ? {signal} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal}) }
  catch (error) { if (signal?.aborted) throw error; throw new ApiError('connection_error', 'Cannot reach the game server. Start the backend, then reconnect.') }
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(data?.error?.code ?? 'server_error', data?.error?.message ?? 'The game server is unavailable. Please try again.')
  return data as T
}
export const api = {
  status: () => request<ModelStatus>('/model/status'),
  daily: () => request<Game>('/game/daily'),
  newGame: (difficulty: Difficulty, retry_game_id?: string) => request<Game>('/game/new', {difficulty, retry_game_id}),
  history: (id: string) => request<Game>(`/game/history/${encodeURIComponent(id)}`),
  move: (game: Game, word: string, power_shot = false) => request<Game>('/game/move', {game_id: game.id, word, expected_move: game.moves_used, request_id: crypto.randomUUID(), power_shot}),
  preview: (game: Game, word: string, power_shot = false, signal?: AbortSignal) => request<MovePreview>('/game/preview', {game_id: game.id, word, expected_move: game.moves_used, power_shot}, signal),
  hint: (game_id: string) => request<Hint>('/game/hint', {game_id}),
}
