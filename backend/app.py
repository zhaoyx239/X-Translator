from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

from .asr_backends import ParaformerASR, SensevoiceASR
from .config import Settings
from .session import TranslationSession
from .visit_counter import VisitCounter
from .zipformer_postprocess import ZipformerTextPostProcessor


class NoCacheStaticFiles(StaticFiles):
    """禁用静态资源缓存，避免前端调试时拿到旧文件。"""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建应用。"""
    app_settings = settings or Settings.from_env()
    app = FastAPI(title="xtranslate", version="0.1.0")
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    app.state.settings = app_settings
    app.state.shared_local_asr_model = None
    app.state.shared_zipformer_postprocessor = None
    app.state.visit_counter = VisitCounter(app_settings.visit_counter_path)

    app.mount("/static", NoCacheStaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        app.state.visit_counter.increment()
        return FileResponse(
            frontend_dir / "index.html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/media/sessions/{session_label}/{filename}")
    async def session_media(session_label: str, filename: str) -> FileResponse:
        return _session_file_response(app_settings.session_audio_dir, session_label, filename)

    @app.get("/api/recent-sessions")
    async def recent_sessions(limit: int = 3) -> JSONResponse:
        items = _collect_recent_sessions(
            sessions_root=app_settings.session_audio_dir,
            limit=max(1, min(limit, 10)),
        )
        return JSONResponse({"items": items})

    @app.get("/api/visit-stats")
    async def visit_stats() -> JSONResponse:
        return JSONResponse({"total_visits": app.state.visit_counter.get_count()})

    @app.get("/api/frontend-config")
    async def frontend_config() -> JSONResponse:
        return JSONResponse({"show_speaker": app_settings.show_speaker})

    @app.on_event("startup")
    async def startup_event() -> None:
        if app_settings.asr_provider == "paraformer":
            app.state.shared_local_asr_model = ParaformerASR(
                device=app_settings.paraformer_device,
                model=app_settings.paraformer_model,
                hub=app_settings.paraformer_hub,
            )
        elif app_settings.asr_provider == "sensevoice":
            app.state.shared_local_asr_model = SensevoiceASR(
                language=app_settings.sensevoice_language,
                device=app_settings.sensevoice_device,
            )
        elif app_settings.asr_provider == "zipformer" and app_settings.zipformer_use_ctpunc:
            app.state.shared_zipformer_postprocessor = ZipformerTextPostProcessor(
                model=app_settings.zipformer_punc_model,
                model_revision=app_settings.zipformer_punc_model_revision,
            )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        session = TranslationSession(
            websocket,
            app_settings,
            shared_local_asr=app.state.shared_local_asr_model,
            zipformer_postprocessor=app.state.shared_zipformer_postprocessor,
        )
        await session.run()

    return app


def _session_file_response(sessions_root: Path, session_label: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"\d{8}_\d{6}_\d{3}", session_label):
        raise HTTPException(status_code=404, detail="audio not found")
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="audio not found")
    session_dir = (sessions_root / session_label).resolve()
    path = (session_dir / filename).resolve()
    if path.parent != session_dir or not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _collect_recent_sessions(sessions_root: Path, limit: int) -> list[dict]:
    if not sessions_root.exists():
        return []
    labels = [
        path.name
        for path in sessions_root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"\d{8}_\d{6}_\d{3}", path.name)
        and (path / "input.wav").exists()
    ]
    labels = sorted(labels, reverse=True)[:limit]
    items = []
    for label in labels:
        input_name = "input.wav"
        output_name = "output.wav"
        output_path = sessions_root / label / output_name
        items.append(
            {
                "label": label,
                "session_audio_name": input_name,
                "session_audio_url": f"/media/sessions/{label}/{input_name}",
                "tts_audio_name": output_name if output_path.exists() else "",
                "tts_audio_url": f"/media/sessions/{label}/{output_name}" if output_path.exists() else "",
            }
        )
    return items
