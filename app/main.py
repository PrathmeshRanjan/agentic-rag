import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.logging import configure_logging
from app.services.audit import init_db

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

load_dotenv()
configure_logging()
init_db()

APP_NAME = os.getenv("APP_NAME", "Agentic HR Chatbot")

app = FastAPI(title=APP_NAME, version="1.0.0")
app.include_router(router)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.is_dir() else None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if templates is None or not (TEMPLATES_DIR / "index.html").is_file():
        return HTMLResponse(f"<h1>{APP_NAME}</h1>")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": APP_NAME},
    )