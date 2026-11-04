"""
FastAPI application for UZEX AI Agent web service.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

from .database import get_database
from .agent import run_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Load .env file automatically
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")

# Global API key from environment / .env file
_api_key: str = os.environ.get("OPENAI_API_KEY", "")
_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing UZEX SQL database...")
    get_database()
    logger.info("UZEX SQL database ready.")
    yield


app = FastAPI(
    title="UZEX AI Agent",
    description="AI-агент для анализа биржевых сделок УЗЕХ / UZEX birja bitimlari tahlili uchun AI-agent",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ------------------------------------------------------------------ #
#  Request / Response models                                           #
# ------------------------------------------------------------------ #

class ConfigureRequest(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    model: str


class StatusResponse(BaseModel):
    status: str
    configured: bool
    model: str
    total_deals: int
    date_range: str


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def root():
    index = static_dir / "index.html"
    return FileResponse(str(index))


@app.post("/configure")
async def configure(req: ConfigureRequest):
    """Set the OpenAI API key and model."""
    global _api_key, _model
    _api_key = req.api_key.strip()
    _model = req.model.strip() or "gpt-4o-mini"
    return {"ok": True, "model": _model}


@app.get("/status", response_model=StatusResponse)
async def status():
    db = get_database()
    stats = db.get_summary_stats()
    return StatusResponse(
        status="ready",
        configured=bool(_api_key),
        model=_model,
        total_deals=stats["total_deals"],
        date_range=stats["date_range"],
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    """Main endpoint: answer a question about UZEX deals."""
    if not _api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    db = get_database()
    try:
        answer = run_agent(
            question=req.question.strip(),
            api_key=_api_key,
            db=db,
            model=_model,
        )
        return AnswerResponse(answer=answer, model=_model)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"ok": True}
