# Semantix API contract

All gameplay lives on the server. Coordinates are rendering-only. JSON keys use snake_case.

`GET /api/game/daily`, `POST /api/game/new`, `POST /api/game/move`, and `GET /api/game/history/{game_id}` return the complete Game below. New accepts `{difficulty, retry_game_id?}` (retry preserves the exact level). Move accepts `{game_id, word, expected_move?, request_id?, power_shot?}`. `expected_move` is the current `moves_used` and prevents racing submissions. Retried request IDs are idempotent.

```ts
type Point = [number, number];
type Game = {
  id: string; level_id: string;
  mode: 'daily' | 'free'; daily_date: string | null;
  difficulty: 'easy' | 'medium' | 'hard'; status: 'playing' | 'won' | 'lost';
  start_word: string; target_word: string;
  move_limit: number; moves_used: number; moves_remaining: number;
  initial_similarity: number; similarity: number; distance: number;
  threshold: number; feedback: string;
  hints_remaining: number; power_shots_remaining: number;
  rating: string | null;
  history: {number: number; word: string; similarity_before: number;
    similarity_after: number; delta: number; efficiency: number;
    position: Point; word_position: Point; power_shot: boolean}[];
  map: {start: Point; target: Point; player: Point; trail: Point[];
    landmarks: {word: string; position: Point}[]};
  physics: {momentum: number; impulse: number; movement_model: string};
};
```

`GET /api/model/status`: `{status: 'unloaded'|'loading'|'ready'|'error', name, dimensions, vocabulary_size, message}`. A new game triggers lazy initialization. `POST /api/model/load` explicitly warms it. Status can also expose `path` and `reduced`.

`POST /api/game/hint` with `{game_id}` returns `{word, remaining}` (non-consuming move, consumes hint budget, easy 3 / medium 1 / hard 0). `GET /api/health` responds without loading the model.

Errors have non-2xx status and `{error: {code, message}}`. Invalid moves do not consume moves. The exact target word cannot be submitted: reach its neighborhood by entering related words. Repeated other words are permitted. UTC defines daily challenge identity; daily results are locally deduplicated by date. Retrying a daily can be practiced but not scored twice.
