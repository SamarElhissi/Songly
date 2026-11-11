import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from songcraft.api.songs import router
from songcraft.config import Settings, get_settings
from songcraft.db import Database
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.runner import run_song
from songcraft.queue import build_job_queue
from songcraft.recovery import recover_interrupted_jobs


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = _resolve(settings or get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        database = Database(resolved)
        database.create_all(resolved)
        app.state.db = database
        app.state.settings = resolved
        app.state.pipeline = build_pipeline(resolved)

        def handle(song_id: str) -> None:
            # Look up the pipeline on each job so a test can swap a step.
            with database.session() as session:
                run_song(song_id, session, app.state.pipeline, resolved)

        with database.session() as session:
            queued_ids = recover_interrupted_jobs(session)
        app.state.queue = build_job_queue(resolved.job_queue, handle)
        app.state.queue.start()
        for song_id in queued_ids:
            app.state.queue.enqueue(song_id)
        try:
            yield
        finally:
            # Stop ACE-Step before joining the queue so an in-flight song unblocks.
            music = getattr(app.state.pipeline, "music", None)
            stop = getattr(music, "shutdown", None)
            if callable(stop):
                stop()
            app.state.queue.shutdown()

    app = FastAPI(title="SongCraft", lifespan=lifespan)
    app.state.settings = resolved
    origins = [item.strip() for item in resolved.cors_origins.split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


def _resolve(settings: Settings) -> Settings:
    return settings.model_copy(update={"data_dir": Path(settings.data_dir).resolve()})


app = create_app()
