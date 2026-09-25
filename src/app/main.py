import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from api.database import init_db, is_configured, seed_defaults

        if not is_configured():
            logger.info("Lakebase is not configured; using built-in presets")
        else:
            init_db()
            seed_defaults()
            logger.info("Lakebase database initialized and seeded")
    except Exception:
        logger.exception("Lakebase initialization failed; using built-in presets")
    yield


app = FastAPI(title="Clinical Extractor", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    index_path = os.path.join(static_dir, "index.html")
    static_root = os.path.realpath(static_dir)

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Resolve the request against the static root and confirm it stays
        # inside it, so crafted paths like "../../etc/passwd" can't escape.
        candidate = os.path.realpath(os.path.join(static_dir, full_path))
        if os.path.isdir(candidate):
            candidate = os.path.realpath(os.path.join(candidate, "index.html"))
        if (
            candidate == static_root or candidate.startswith(static_root + os.sep)
        ) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(index_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
