export type Point = [number, number]
export type Difficulty = 'easy' | 'medium' | 'hard'
export type Mode = 'daily' | 'free'
export interface Move {
  number: number; word: string; similarity_before: number; similarity_after: number
  delta: number; efficiency: number; position: Point; word_position: Point; power_shot: boolean
}
export interface Game {
  id: string; level_id: string; mode: Mode; daily_date: string | null
  difficulty: Difficulty; status: 'playing' | 'won' | 'lost'; start_word: string; target_word: string
  move_limit: number; moves_used: number; moves_remaining: number; initial_similarity: number
  similarity: number; distance: number; threshold: number; feedback: string
  hints_remaining: number; power_shots_remaining: number; rating: string | null; history: Move[]
  map: { start: Point; target: Point; player: Point; trail: Point[]; landmarks: { word: string; position: Point }[] }
  physics: {momentum: number; impulse: number; movement_model: string}
}
export interface ModelStatus {
  status: 'unloaded' | 'loading' | 'ready' | 'error'; name: string; dimensions: number; vocabulary_size: number; message: string
}
export interface CompletedGame {
  id: string; daily_date: string | null; mode: Mode; status: 'won' | 'lost'; moves: number; similarity: number; rating: string | null; completed_at: string
}
