"""
View routes for serving HTML templates.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..core.config import TEMPLATES_DIR

router = APIRouter(tags=["Views"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})
