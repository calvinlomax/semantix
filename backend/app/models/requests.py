from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NewGameRequest(RequestModel):
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    retry_game_id: str | None = Field(default=None, min_length=1, max_length=100)


class MoveRequest(RequestModel):
    game_id: str = Field(min_length=1, max_length=100)
    word: str = Field(min_length=1, max_length=64)
    expected_move: int | None = Field(default=None, ge=0, le=100)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    power_shot: bool = False


class HintRequest(RequestModel):
    game_id: str = Field(min_length=1, max_length=100)
