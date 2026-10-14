# Semantix API contract

All gameplay lives on the server. Coordinates are rendering-only. JSON keys use snake_case.

`GET /api/game/daily`, `POST /api/game/new`, `POST /api/game/move`, and `GET /api/game/history/{game_id}` return the complete Game below. New accepts `{difficulty, retry_game_id?}` (retry preserves the exact level). Move accepts `{game_id, word, expected_move?, request_id?, power_shot?}`. `expected_move` is the current `moves_used` and prevents racing submissions. Retried request IDs are idempotent.

```ts
type Point = [number, number];
type Point3D = [number, number, number];
type Resolution = {
  word: string; // Original submitted spelling, with surrounding whitespace trimmed.
  resolved_word: string; // Model entry actually used; never a fuzzy spelling match.
  resolution: 'exact' | 'normalized' | 'lemma';
  normalization_message: string | null;
};
type Move = Resolution & {
  number: number; similarity_before: number; similarity_after: number;
  word_similarity: number; // Entered word's cosine similarity to the target.
  delta: number; efficiency: number; power_shot: boolean;
  position: Point; word_position: Point;
  position3d: Point3D; word_position3d: Point3D;
};
type MapSpace<T> = {
  start: T; target: T; player: T; trail: T[];
  landmarks: {word: string; position: T; similarity: number}[];
};
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
  history: Move[];
  map: MapSpace<Point>;
  map3d: MapSpace<Point3D>;
  physics: {momentum: number; impulse: number; movement_model: string};
};
```

`map` remains compatible with existing 2D clients. `map3d` uses the same fixed PCA transform, retaining a third component. Start, target, and landmarks remain fixed throughout a round; each successful move appends a landing position to both trails. Every `position` is the first two coordinates of its corresponding `position3d`. All scores and movement are calculated using 300-dimensional embeddings before projection. Older completed games in browser storage may lack `map3d` and resolution metadata; the UI renders them in 2D.

`POST /api/game/preview` accepts `{game_id, word, expected_move?, power_shot?}` and returns:

```ts
type MovePreview = Move & {
  game_id: string;
  expected_move: number; // Current committed moves_used, not the predicted number.
  would_win: boolean;
};
```

Preview uses the same validation, resolution, physics, and scoring as submission. It does not change position, velocity, history, repetition tracking, move/hint/power budgets, status, or idempotency records. A subsequent submission with the same state, word, and power setting produces identical scores and coordinates. Include `expected_move` to reject a stale preview or submission with HTTP 409. Preview cannot accept client-supplied positions or velocities; unexpected request fields receive HTTP 422. The frontend debounces previews for 350 ms and discards responses after input or state changes. Daily previews default off; Free Play previews default on.

`GET /api/model/status` reports guess-model entries separately from puzzle words:

```ts
type ModelStatus = {
  status: 'unloaded' | 'loading' | 'ready' | 'error';
  name: string; dimensions: 300; message: string;
  vocabulary_size: number;         // Exact entries loaded from the reduced .kv.
  target_vocabulary_size: number;  // Curated level words actually retained.
  requested_guess_count: number | null; // From the preparation manifest.
  missing_guess_count: number;
  requested_target_count: number;
  reduced: boolean | null;
  memory_mapped: boolean;
};
```

A new game triggers lazy initialization; `POST /api/model/load` explicitly warms it. Before loading, counts are zero or null. The current prepared model reports 73,604 entries, 2,674 retained targets, 94,037 requested guesses, and 20,433 missing guess candidates. The count of resolvable submitted forms can be larger because of normalization and lemmatization. Runtime requires the reduced native `.kv`, its vector sidecar, and its manifest; original Google News binaries/full-model artifacts produce a structured `model_unavailable` error.

`POST /api/game/hint` with `{game_id}` returns `Resolution & {remaining: number}` (non-consuming move, consumes hint budget, easy 3 / medium 1 / hard 0). Hints must pass the same word/rule validation before consuming a hint. `GET /api/health` responds without loading the model.

Validation normalizes the token, rejects anything outside one alphabetic word of 2–24 letters, and checks blocked tokens/families. Lookup then prefers the exact original model entry, then the normalized lowercase entry, then an unambiguous WordNet lemma available in the model. `normalization_message` is, for example, `"Prepared" accepted using "prepared"`. History always preserves `word`; rendering and physics use the resolved embedding. Ambiguous missing forms such as `better` (good/well) are rejected with an explanation instead of selecting a meaning based on vocabulary availability. Exact `better` is accepted when retained. No synonyms or fuzzy spelling matches are used.

`allowed_guesses.txt` supplies offline model preparation; it is not a runtime allowlist. A listed source word missing from the `.kv` is unavailable unless normalization/lemmatization resolves to another retained entry. Conversely, a form absent from the text file can resolve to a retained model entry. Puzzle starts/targets must belong to `level_words.txt` and have exact retained embeddings. Increasing the guess vocabulary does not expand the target pool.

Target entry is rejected both directly and when a fallback resolves to the target model entry. Exact inflections with distinct embeddings retain their prior legality. Forbidden-word rules and optional `allow_repeats: false` mechanics compare grammatical families, so inflections cannot bypass those restrictions. Standard levels allow repeated non-target words. These rules apply to hints, previews, and committed moves.

Errors have non-2xx status and `{error: {code, message}}`. Relevant codes include `unknown_word`, `ambiguous_word`, `invalid_word`, `target_word`, `forbidden_word`, `repeated_word`, `forbidden_region`, `stale_game`, and `normalization_unavailable` (HTTP 503, local WordNet setup required). Invalid moves and previews spend no move or power budget. Reusing a move `request_id` with a different submitted word/case or power setting produces `request_conflict`; an identical retry returns the saved response.

UTC defines daily challenge identity; daily results are locally deduplicated by date. Retrying a daily can be practiced but not scored twice.

Semantic feedback additions
--------------------------

Both move-history entries and move previews include `semantic`:

```json
{
  "rank": 412,
  "vocabulary_size": 73604,
  "percentile": 99.4416,
  "bucket": "burning",
  "versus_best": "warmer",
  "versus_previous": "warmer",
  "same_family": false,
  "cluster": "same cluster",
  "large_semantic_jump": false,
  "previous_guess_similarity": 0.52
}
```

Ranks are 1-based over validated prepared-model entries, target included; equal
cosines share rank. Percentile is `100 * (N - rank) / max(1, N - 1)`.
`versus_best` and `versus_previous` are `first | warmer | colder | same`.
`cluster` is `first | same cluster | new semantic cluster`; these are cosine
heuristics documented in README, not taxonomy. Previous similarity is null on
the first guess. Preview computes these fields without changing history or
repetition state. `word_similarity` remains the raw word-to-target cosine.
The existing position similarity, movement, win thresholds, and response fields
are unchanged. History retains original submitted spelling.

Model status adds `hint_vocabulary_size`: the number of safe display words
retained in the model. Rank population can be smaller than model vocabulary size
when a deployment adds blocked families after preparation.
