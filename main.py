
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import init_db
from app.agents.agent_loop import get_agent_runner
from app.tools.search_catalog import get_catalog_searcher
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup code before the server starts accepting requests,
    and cleanup code when the server shuts down.

    Startup tasks (happen ONCE when the server starts):
      1. Initialize database (create tables if missing)
      2. Warm up the HybridCatalogSearcher (load ~80MB model, init ChromaDB)
      3. Warm up the SalesAgentRunner (initialize LangGraph + LLM)

    WHY WARM UP?
      Loading the embedding model and ChromaDB takes 5-15 seconds.
      If we defer this to the first request, that user gets a very slow response.
      By warming up at startup, the first real request is just as fast as any other.
    """
    logger.info("=" * 60)
    logger.info("Starting SaaSify Sales Agent API...")
    logger.info("=" * 60)

    logger.info("Step 1/3: Initializing database...")
    init_db()
    logger.info("Database ready.")

    logger.info("Step 2/3: Warming up catalog searcher (this may take ~15s first run)...")
    get_catalog_searcher()
    logger.info("Catalog searcher ready.")

    logger.info("Step 3/3: Initializing agent runner...")
    get_agent_runner()
    logger.info("Agent runner ready.")

    logger.info("=" * 60)
    logger.info("All systems ready. Server is accepting requests.")
    logger.info(f"Model: {settings.LLM_MODEL}")
    logger.info(f"Database: {settings.DATABASE_URL}")
    logger.info("Docs available at: /docs")
    logger.info("=" * 60)

    yield  # Server runs while we're here

    logger.info("Server shutting down.")


app = FastAPI(
    title="SaaSify Sales Assistant Agent",
    description=(
        "A persistent B2B SaaS sales assistant with cross-session memory, "
        "tool use (hybrid search), and self-evaluation on every response."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # All origins (restrict in production!)
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, DELETE, etc.
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["System"])
def root():
    """Root endpoint — confirms the API is running."""
    return {
        "service": "SaaSify Sales Assistant Agent",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,   # Set to True during local development for auto-reload
        log_level="info",
    )
