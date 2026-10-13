"""Single-process store; replace this interface with Redis for multiple workers."""
from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Any

from .errors import GameError


class GameStore:
    def __init__(self, capacity: int = 1000, ttl: float = 86400, clock=monotonic):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._games: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()

    def _prune(self):
        now = self.clock()
        expired = [key for key, (created, _) in self._games.items() if now - created >= self.ttl]
        for key in expired:
            del self._games[key]

    def put(self, game):
        with self._lock:
            self._prune()
            self._games[game.id] = (self.clock(), game)
            while len(self._games) > self.capacity:
                self._games.popitem(last=False)

    def get(self, game_id: str):
        with self._lock:
            self._prune()
            entry = self._games.get(game_id)
            if entry is None:
                raise GameError("game_expired", "This game has expired or the server restarted. Start a new round.", 404)
            self._games.move_to_end(game_id)
            return entry[1]
