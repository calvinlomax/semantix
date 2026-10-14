from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from secrets import choice, token_urlsafe
from threading import RLock

import numpy as np

from ..config import Settings
from ..engine.embeddings import EmbeddingStore, WordValidationError, clean_word
from ..engine.level_generator import Level, LevelGenerator
from ..engine.semantic import SemanticIndex
from ..engine.physics import PhysicsConfig, step
from ..engine.projection import Projection
from ..engine.scoring import cosine_similarity, efficiency, feedback
from .errors import GameError
from .store import GameStore

logger = logging.getLogger(__name__)


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def daily_index(day: str, size: int) -> int:
    return int.from_bytes(sha256(f"semantix-v1:{day}".encode()).digest()[:8], "big") % size


@dataclass
class Session:
    id: str
    level: Level
    mode: str
    daily_date: str | None
    position: np.ndarray
    velocity: np.ndarray
    projection: Projection
    landmarks: list[dict]
    trail: list[list[float]]
    trail3d: list[list[float]]
    history: list[dict] = field(default_factory=list)
    status: str = "playing"
    hints_used: int = 0
    power_used: int = 0
    last_word: str | None = None
    used_word_keys: set[str] = field(default_factory=set)
    requests: dict = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)


class GameService:
    def __init__(self, settings: Settings, embeddings=None, levels=None, day_provider=utc_day):
        self.settings = settings
        self.embeddings = embeddings if embeddings is not None else EmbeddingStore(
            settings.model_path, settings.vocabulary_path, denylist_path=settings.blocked_path,
            target_vocabulary_path=settings.target_vocabulary_path, hint_vocabulary_path=settings.hint_vocabulary_path, require_prepared=True,
            binary=settings.model_binary, limit=settings.model_limit,
        )
        self.semantic = SemanticIndex(self.embeddings)
        self.physics = PhysicsConfig(momentum=settings.momentum, impulse=settings.impulse, movement_model=settings.movement_model)
        self.store = GameStore(settings.max_games, settings.game_ttl)
        self.levels = levels
        self.generator = None
        self.day_provider = day_provider
        self._lock = RLock()
        self._state = "ready" if levels is not None else "unloaded"
        self._message = "Ready to explore." if levels is not None else "The model loads when you start your first round."

    def model_status(self):
        details = self.embeddings.status()
        return {
            "status": self._state, "name": "Google News Word2Vec", "dimensions": 300,
            "vocabulary_size": details.get("vocabulary_size", 0),
            "target_vocabulary_size": details.get("target_vocabulary_size", details.get("curated_words", 0)),
            "hint_vocabulary_size": details.get("hint_vocabulary_size", len(self.embeddings.hint_words) if details.get("ready") else 0),
            "requested_guess_count": details.get("requested_guess_count"),
            "missing_guess_count": details.get("missing_guess_count", 0),
            "requested_target_count": details.get("requested_target_count"),
            "reduced": details.get("reduced"), "memory_mapped": details.get("memory_mapped", False),
            "message": self._message,
        }

    def ensure_ready(self):
        with self._lock:
            if self._state == "ready":
                if self.generator is None:
                    self.generator = LevelGenerator(self.embeddings, self.physics)
                return
            self._state = "loading"
            self._message = "Loading Google News vectors and validating the puzzle catalog…"
            try:
                self.embeddings.load()
                self.generator = LevelGenerator(self.embeddings, self.physics)
                if self.levels is None:
                    self.levels = self._load_levels()
                if not self.levels:
                    raise ValueError("No validated levels are available. Run the level-generation command in README.md.")
                self._state = "ready"
                size = self.embeddings.status()["vocabulary_size"]
                self._message = f"Ready · {size:,} guess embeddings · {len(self.embeddings.words):,} curated puzzle words · 300 dimensions."
            except Exception as exc:
                self._state = "error"
                self._message = str(exc)
                logger.exception("Model initialization failed")
                raise GameError("model_unavailable", f"Unable to prepare the game: {exc}. See README.md for the model preparation commands.", 503) from exc

    def _load_levels(self):
        path = self.settings.levels_path
        if path.exists():
            catalog = json.loads(path.read_text())
            if catalog.get("embedding_fingerprint") != self.embeddings.fingerprint:
                raise ValueError("The level catalog belongs to different model vectors. Regenerate levels for this model")
            if catalog.get("physics") != asdict(self.physics):
                raise ValueError("The level catalog uses different movement settings. Regenerate levels with the current settings")
            levels = [Level.from_dict(value) for value in catalog["levels"]]
            if not all(self.generator.validate(level) for level in levels):
                raise ValueError("The catalog contains an invalid or unreachable level. Regenerate levels")
            return levels
        # Useful for a first launch; persisted catalogs are recommended for fixed daily identity.
        return self.generator.generate(count_per_difficulty=3, seed=42)

    def new_game(self, difficulty="medium", mode="free", retry_game_id=None):
        self.ensure_ready()
        day = None
        if retry_game_id:
            old = self.store.get(retry_game_id)
            level, mode, day = old.level, old.mode, old.daily_date
        else:
            candidates = sorted((item for item in self.levels if item.difficulty == ("medium" if mode == "daily" else difficulty)), key=lambda x: x.id)
            if not candidates:
                raise GameError("no_levels", "No validated levels exist for this difficulty. Regenerate the level catalog.", 503)
            if mode == "daily":
                day = self.day_provider()
                level = candidates[daily_index(day, len(candidates))]
            else:
                level = choice(candidates)
        start = self.embeddings.lookup(level.start_word)
        target = self.embeddings.lookup(level.target_word)
        landmark_words = list(dict.fromkeys([level.start_word, level.target_word, *(word for word in level.landmark_words if word in self.semantic.hint_words)]))
        anchors = np.stack([self.embeddings.lookup(word) for word in landmark_words])
        projection = Projection.fit(anchors)
        position = start.copy()
        game = Session(
            id=token_urlsafe(24), level=level, mode=mode, daily_date=day,
            position=position, velocity=np.zeros(300), projection=projection,
            landmarks=[{"word": word, "position": projection.project(self.embeddings.lookup(word)),
                        "position3d": projection.project3d(self.embeddings.lookup(word)),
                        "similarity": cosine_similarity(self.embeddings.lookup(word), target)}
                       for word in landmark_words if word not in (level.start_word, level.target_word)],
            trail=[projection.project(position)], trail3d=[projection.project3d(position)], last_word=level.start_word,
        )
        self.store.put(game)
        return self.serialize(game)

    def _hints_limit(self, game):
        return {"easy": 3, "medium": 1, "hard": 0}[game.level.difficulty]

    def _physics_for(self, game):
        mechanics = getattr(game.level, "mechanics", {}) or {}
        changes = {key: mechanics[key] for key in ("gravity_strength", "power_multiplier", "forbidden_similarity") if key in mechanics}
        return PhysicsConfig(**{**asdict(self.physics), **changes})

    def _resolve_allowed(self, game, word):
        """One authoritative vocabulary/rules path for previews, hints, and moves."""
        try:
            resolved = self.embeddings.resolve(word)
            keys = resolved.rule_keys
            # Keep the existing target rule tied to the actual model entry.
            # Distinct exact inflections have distinct vectors and already form
            # valid catalog routes; a fallback resolving TO the target may not
            # smuggle that target vector into play.
            target_token = clean_word(game.level.target_word)
            if target_token in {clean_word(resolved.original_word), clean_word(resolved.resolved_word)}:
                raise GameError("target_word", "Navigate with related words; the target or a word resolved to the target cannot be used.")
            mechanics = getattr(game.level, "mechanics", {}) or {}
            for forbidden in mechanics.get("forbidden_words", []):
                if keys & self.embeddings.rule_keys(forbidden):
                    raise GameError("forbidden_word", "That word or its base form is forbidden in this level.")
            if not mechanics.get("allow_repeats", True) and keys & game.used_word_keys:
                raise GameError("repeated_word", "You have already used this word or one of its grammatical forms.")
            config = self._physics_for(game)
            for region in mechanics.get("forbidden_regions", []):
                if cosine_similarity(resolved.vector, self.embeddings.lookup(region)) >= config.forbidden_similarity:
                    raise GameError("forbidden_region", "That word is inside a forbidden semantic neighborhood.")
            return resolved
        except WordValidationError as exc:
            raise GameError(exc.code, str(exc), 503 if exc.code == "normalization_unavailable" else 400) from exc

    @staticmethod
    def _resolution_fields(resolved):
        return {
            "word": resolved.original_word, "resolved_word": resolved.resolved_word,
            "resolution": resolved.resolution,
            "normalization_message": (f'"{resolved.original_word}" accepted using "{resolved.resolved_word}"'
                                      if resolved.original_word != resolved.resolved_word else None),
        }

    def _calculate_move(self, game, request):
        """Calculate a validated update without changing any session field."""
        if game.status != "playing":
            raise GameError("game_complete", "This round is complete. Start a new round or retry.", 409)
        if request.expected_move is not None and request.expected_move != len(game.history):
            raise GameError("stale_game", "Another move was already processed. Refresh this round before submitting again.", 409)
        resolved = self._resolve_allowed(game, request.word)
        mechanics = getattr(game.level, "mechanics", {}) or {}
        if request.power_shot and game.power_used >= int(mechanics.get("power_shots", 0)):
            raise GameError("no_power_shots", "No power shots are available in this round.")
        target = self.embeddings.lookup(game.level.target_word)
        before = cosine_similarity(game.position, target)
        position, velocity = step(
            game.position, game.velocity, resolved.vector, self._physics_for(game),
            previous_word=self.embeddings.lookup(game.last_word),
            attractor=self.embeddings.lookup(mechanics["attractor"]) if mechanics.get("attractor") else None,
            power=request.power_shot,
        )
        after = cosine_similarity(position, target)
        entry = {
            "number": len(game.history) + 1, **self._resolution_fields(resolved),
            "similarity_before": before, "similarity_after": after, "delta": after - before,
            "word_similarity": cosine_similarity(resolved.vector, target),
            "semantic": self.semantic.feedback(game.level.target_word, resolved, game.history, game.used_word_keys),
            "efficiency": efficiency(before, after, game.level.threshold),
            "position": game.projection.project(position), "word_position": game.projection.project(resolved.vector),
            "position3d": game.projection.project3d(position), "word_position3d": game.projection.project3d(resolved.vector),
            "power_shot": request.power_shot,
        }
        return resolved, position, velocity, entry

    def preview(self, request):
        game = self.store.get(request.game_id)
        with game.lock:
            _, _, _, entry = self._calculate_move(game, request)
            return {"game_id": game.id, "expected_move": len(game.history), **entry,
                    "would_win": entry["similarity_after"] >= game.level.threshold}

    def move(self, request):
        game = self.store.get(request.game_id)
        with game.lock:
            # Preserve casing: distinct exact-case model entries can have distinct vectors.
            signature = (request.word.strip(), request.power_shot)
            if request.request_id and request.request_id in game.requests:
                old_signature, response = game.requests[request.request_id]
                if signature != old_signature:
                    raise GameError("request_conflict", "A request ID cannot be reused for another word.", 409)
                return response
            resolved, position, velocity, entry = self._calculate_move(game, request)
            game.position, game.velocity, game.last_word = position, velocity, resolved.resolved_word
            game.history.append(entry)
            game.trail.append(entry["position"])
            game.trail3d.append(entry["position3d"])
            game.used_word_keys.update(resolved.rule_keys)
            game.power_used += int(request.power_shot)
            if entry["similarity_after"] >= game.level.threshold:
                game.status = "won"
            elif len(game.history) >= game.level.move_limit:
                game.status = "lost"
            response = self.serialize(game)
            if request.request_id:
                game.requests[request.request_id] = (signature, response)
            return response

    def history(self, game_id):
        game = self.store.get(game_id)
        with game.lock:
            return self.serialize(game)

    def hint(self, game_id):
        game = self.store.get(game_id)
        with game.lock:
            if game.status != "playing":
                raise GameError("game_complete", "This round is complete.", 409)
            if game.hints_used >= self._hints_limit(game):
                raise GameError("no_hints", "No hints remain for this round.")
            target = self.embeddings.lookup(game.level.target_word)
            resolved, best_score = None, -float("inf")
            for candidate, _ in self.semantic.neighbors(game.level.target_word, 128):
                try:
                    suggestion = self._resolve_allowed(game, candidate)
                except GameError as exc:
                    if exc.code == "normalization_unavailable":
                        raise
                    continue
                mechanics = getattr(game.level, "mechanics", {}) or {}
                predicted, _ = step(game.position, game.velocity, suggestion.vector, self._physics_for(game),
                                    previous_word=self.embeddings.lookup(game.last_word),
                                    attractor=self.embeddings.lookup(mechanics["attractor"]) if mechanics.get("attractor") else None)
                score = cosine_similarity(predicted, target)
                if score > best_score:
                    best_score, resolved = score, suggestion
            if resolved is None:
                raise GameError("no_hint_available", "No legal safe hint is available from this position.")
            game.hints_used += 1
            return {**self._resolution_fields(resolved), "remaining": self._hints_limit(game) - game.hints_used}

    def serialize(self, game):
        level = game.level
        similarity = cosine_similarity(game.position, self.embeddings.lookup(level.target_word))
        used = len(game.history)
        mechanics = getattr(level, "mechanics", {}) or {}
        rating = None
        if game.status == "won":
            rating = "Brilliant" if used <= 3 else "Well navigated" if used < level.move_limit else "Just in orbit"
        elif game.status == "lost":
            rating = "Keep exploring"
        return {
            "id": game.id, "level_id": level.id, "mode": game.mode, "daily_date": game.daily_date,
            "difficulty": level.difficulty, "status": game.status, "start_word": level.start_word,
            "target_word": level.target_word, "move_limit": level.move_limit, "moves_used": used,
            "moves_remaining": max(0, level.move_limit - used), "initial_similarity": level.initial_similarity,
            "similarity": similarity, "distance": 1.0 - similarity, "threshold": level.threshold,
            "feedback": feedback(similarity, level.threshold, level.feedback_bands),
            "hints_remaining": max(0, self._hints_limit(game) - game.hints_used),
            "power_shots_remaining": max(0, int(mechanics.get("power_shots", 0)) - game.power_used),
            "rating": rating, "history": list(game.history),
            "map": {"start": game.trail[0], "target": game.projection.project(self.embeddings.lookup(level.target_word)),
                    "player": game.projection.project(game.position), "trail": list(game.trail),
                    "landmarks": [{"word": item["word"], "position": item["position"], "similarity": item["similarity"]} for item in game.landmarks]},
            "map3d": {"start": game.trail3d[0], "target": game.projection.project3d(self.embeddings.lookup(level.target_word)),
                      "player": game.projection.project3d(game.position), "trail": list(game.trail3d),
                      "landmarks": [{"word": item["word"], "position": item["position3d"], "similarity": item["similarity"]} for item in game.landmarks]},
            "physics": {"momentum": self.physics.momentum, "impulse": self.physics.impulse, "movement_model": self.physics.movement_model},
        }
