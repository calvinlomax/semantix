export type Point = [number, number]
export type Point3D = [number, number, number]
export type Difficulty = 'easy' | 'medium' | 'hard'
export type Mode = 'daily' | 'free'
export interface WordResolution {
  resolved_word: string; resolution: string; normalization_message: string | null
}
export interface MapSpace<T> {
  start: T; target: T; player: T; trail: T[]; landmarks: {word: string; position: T; similarity?: number}[]
}
export interface Move {
  number: number; word: string; similarity_before: number; similarity_after: number
  delta: number; efficiency: number; position: Point; word_position: Point; power_shot: boolean
  resolved_word?: string; resolution?: string; normalization_message?: string | null
  position3d?: Point3D; word_position3d?: Point3D
  word_similarity?: number
}
export interface MovePreview extends WordResolution {
  game_id: string; expected_move: number; word: string; position: Point; word_position: Point
  position3d: Point3D; word_position3d: Point3D; similarity_before: number; similarity_after: number
  delta: number; efficiency: number; would_win: boolean; power_shot: boolean
  word_similarity: number
}
export interface Hint extends WordResolution {word: string; remaining: number}
export interface Game {
  id: string; level_id: string; mode: Mode; daily_date: string | null
  difficulty: Difficulty; status: 'playing' | 'won' | 'lost'; start_word: string; target_word: string
  move_limit: number; moves_used: number; moves_remaining: number; initial_similarity: number
  similarity: number; distance: number; threshold: number; feedback: string
  hints_remaining: number; power_shots_remaining: number; rating: string | null; history: Move[]
  map: MapSpace<Point>
  map3d?: MapSpace<Point3D>
  physics: {momentum: number; impulse: number; movement_model: string}
}
export interface ModelStatus {
  status: 'unloaded' | 'loading' | 'ready' | 'error'; name: string; dimensions: number; vocabulary_size: number; message: string
  target_vocabulary_size?: number; requested_guess_count?: number | null; missing_guess_count?: number
  requested_target_count?: number | null; reduced?: boolean | null; memory_mapped?: boolean
}
export interface CompletedGame {
  id: string; daily_date: string | null; mode: Mode; status: 'won' | 'lost'; moves: number; similarity: number; rating: string | null; completed_at: string
}
