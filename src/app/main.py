import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from databricks.sdk import WorkspaceClient
from dbdemos_tracker import Tracker
from api.routes import router
from api.evaluations import router as evaluations_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.usage_tracker = None
    if os.environ.get("DATABRICKS_HOST"):
        try:
            client = WorkspaceClient(http_timeout_seconds=2, retry_timeout_seconds=0)
            _app.state.usage_tracker = Tracker(
                org_id=client.get_workspace_id(),
                demo_name="hls-clinical-extractor",
            )
        except Exception:
            logger.debug("Optional usage tracking initialization failed", exc_info=True)

    try:
        from api.database import (
            init_db,
            is_configured,
            mark_evaluation_runs_interrupted,
            seed_defaults,
        )

        if not is_configured():
            logger.info("Lakebase is not configured; using built-in presets")
        else:
            init_db()
            seed_defaults()
            interrupted = mark_evaluation_runs_interrupted()
            if interrupted:
                logger.info("Marked %d unfinished evaluation runs as interrupted", interrupted)
            logger.info("Lakebase database initialized and seeded")
    except Exception:
        logger.exception("Lakebase initialization failed; using built-in presets")
    yield


app = FastAPI(title="Clinical Extractor", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
app.include_router(evaluations_router, prefix="/api/v1")


@app.middleware("http")
async def track_app_view(request: Request, call_next):
    # The library's generic middleware also emits anonymous events for other domains.
    tracker = getattr(request.app.state, "usage_tracker", None)
    email = request.headers.get("x-forwarded-email", "").strip().casefold()
    path = request.url.path
    if (
        tracker is not None
        and request.method == "GET"
        and path in {"/", "/index.html", "/info/", "/info/index.html"}
        and email.endswith("@databricks.com")
    ):
        try:
            tracker.track_app_view(user_email=email, app_path=path)
        except Exception:
            logger.debug("Optional usage tracking failed", exc_info=True)
    return await call_next(request)


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    index_path = os.path.join(static_dir, "index.html")
    static_root = os.path.realpath(static_dir)

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        # Resolve the request against the static root and confirm it stays
        # inside it, so crafted paths like "../../etc/passwd" can't escape.
        candidate = os.path.realpath(os.path.join(static_dir, full_path))
        is_directory = os.path.isdir(candidate)
        if is_directory:
            candidate = os.path.realpath(os.path.join(candidate, "index.html"))
        if (
            candidate == static_root or candidate.startswith(static_root + os.sep)
        ) and os.path.isfile(candidate):
            # A trailing slash lets directory pages use portable relative assets.
            if is_directory and not request.url.path.endswith("/"):
                return RedirectResponse(request.url.replace(path=request.url.path + "/"))
            return FileResponse(candidate)
        return FileResponse(index_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
