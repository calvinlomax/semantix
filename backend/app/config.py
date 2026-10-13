from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=lambda: project_path(os.getenv("SEMANTIX_MODEL_PATH", "data/models/google-news-300.kv")))
    vocabulary_path: Path = field(default_factory=lambda: project_path(os.getenv("SEMANTIX_VOCABULARY_PATH", "data/vocabulary/words.txt")))
    blocked_path: Path = field(default_factory=lambda: project_path(os.getenv("SEMANTIX_BLOCKED_PATH", "data/vocabulary/blocked.txt")))
    levels_path: Path = field(default_factory=lambda: project_path(os.getenv("SEMANTIX_LEVELS_PATH", "data/levels/catalog.json")))
    model_binary: bool = field(default_factory=lambda: os.getenv("SEMANTIX_MODEL_BINARY", "false").lower() == "true")
    model_limit: int | None = field(default_factory=lambda: int(os.environ["SEMANTIX_MODEL_LIMIT"]) if os.getenv("SEMANTIX_MODEL_LIMIT") else None)
    momentum: float = field(default_factory=lambda: float(os.getenv("SEMANTIX_MOMENTUM", ".35")))
    impulse: float = field(default_factory=lambda: float(os.getenv("SEMANTIX_IMPULSE", ".45")))
    movement_model: str = field(default_factory=lambda: os.getenv("SEMANTIX_MOVEMENT_MODEL", "direction"))
    max_games: int = field(default_factory=lambda: int(os.getenv("SEMANTIX_MAX_GAMES", "1000")))
    game_ttl: float = field(default_factory=lambda: float(os.getenv("SEMANTIX_GAME_TTL", "86400")))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(os.getenv("SEMANTIX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")))

    def __post_init__(self):
        if not 0 <= self.momentum < 1 or not 0 < self.impulse <= 2:
            raise ValueError("Momentum must be in [0,1), impulse in (0,2].")
        if self.max_games < 1 or self.game_ttl <= 0:
            raise ValueError("Game storage bounds must be positive.")
