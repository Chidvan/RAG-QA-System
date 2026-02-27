"""FastAPI application entry point."""

# IMPORTANT: Load .env file FIRST
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import documents, health, query
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

# Load settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Log level: {settings.log_level}")

    yield

    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## RAG Q&A System API

A Retrieval-Augmented Generation (RAG) question-answering system built with:
- FastAPI for API layer
- LangChain for RAG orchestration
- Qdrant for vector storage
- Gemini / OpenAI models for LLM

### Features
- Upload PDF, TXT, CSV
- Ask AI-powered questions
- Source document transparency
- Streaming responses
""",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static Files (CI-Safe Mount)
# -----------------------------

STATIC_DIR = Path("static")

if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------------
# Routers
# -----------------------------

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)

# -----------------------------
# Root Endpoint (Safe)
# -----------------------------

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve UI if available; otherwise show API status."""
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return index_file.read_text()

    return HTMLResponse(
        content="<h3>RAG Q&A System API is running</h3>",
        status_code=200,
    )

# -----------------------------
# Global Exception Handler
# -----------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )

# -----------------------------
# Local Development Entry
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )