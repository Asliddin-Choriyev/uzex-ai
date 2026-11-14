"""
FastAPI application for UZEX AI Agent web service.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Model config (read once at startup is fine for model name)
_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# _api_key is intentionally NOT cached at module load time.
# Vercel injects env vars before each function invocation, so we
# read it fresh inside every request handler to be safe.
def _get_api_key() -> str:
    """Return the OpenAI API key from environment (checked fresh each call)."""
    return os.environ.get("OPENAI_API_KEY", "")


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
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UZEX AI Agent API is running</h1>")


@app.post("/configure")
async def configure(req: ConfigureRequest):
    """Set the OpenAI API key and model (overrides env var for this process)."""
    global _model
    # Store in environment so _get_api_key() picks it up
    os.environ["OPENAI_API_KEY"] = req.api_key.strip()
    _model = req.model.strip() or "gpt-4o-mini"
    os.environ["OPENAI_MODEL"] = _model
    return {"ok": True, "model": _model}


@app.get("/status", response_model=StatusResponse)
async def status():
    db = get_database()
    stats = db.get_summary_stats()
    return StatusResponse(
        status="ready",
        configured=bool(_get_api_key()),
        model=_model,
        total_deals=stats["total_deals"],
        date_range=stats["date_range"],
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    """Main endpoint: answer a question about UZEX deals."""
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable in Vercel.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    db = get_database()
    try:
        answer = run_agent(
            question=req.question.strip(),
            api_key=api_key,
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
