"""Run from the repository root: uvicorn backend.app.main:app --reload."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .models.requests import HintRequest, MoveRequest, NewGameRequest, PreviewRequest
from .services.errors import GameError
from .services.game import GameService


def create_app(service: GameService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    application = FastAPI(title="Semantix", version="1.0.0", description="Authoritative 300-dimensional word navigation.")
    application.state.game_service = service or GameService(settings)
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.middleware("http")
    async def prevent_session_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(GameError)
    async def game_error_handler(request: Request, exc: GameError):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_input", "message": "Please provide a valid word and game identifier; only documented input fields are accepted."}})

    @application.get("/api/health")
    def health():
        return {"status": "ok", "service": "semantix", "version": "1.0.0"}

    @application.get("/api/model/status")
    def model_status():
        return application.state.game_service.model_status()

    @application.post("/api/model/load")
    def load_model():
        application.state.game_service.ensure_ready()
        return application.state.game_service.model_status()

    @application.get("/api/game/daily")
    def daily():
        return application.state.game_service.new_game(mode="daily")

    @application.post("/api/game/new")
    def new_game(payload: NewGameRequest):
        return application.state.game_service.new_game(payload.difficulty, retry_game_id=payload.retry_game_id)

    @application.post("/api/game/move")
    def move(payload: MoveRequest):
        return application.state.game_service.move(payload)

    @application.post("/api/game/preview")
    def preview(payload: PreviewRequest):
        return application.state.game_service.preview(payload)

    @application.get("/api/game/history/{game_id}")
    def history(game_id: str):
        return application.state.game_service.history(game_id)

    @application.post("/api/game/hint")
    def hint(payload: HintRequest):
        return application.state.game_service.hint(payload.game_id)

    return application


app = create_app()
