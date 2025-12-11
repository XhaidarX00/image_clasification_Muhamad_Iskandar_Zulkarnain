"""
FastAPI Application Entry Point
Cat vs Dog Image Classification
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .core.config import STATIC_DIR, ensure_directories
from .core.ai_model import ai_model
from .api.endpoints import router as api_router
from .api.views import router as views_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting application...")
    ensure_directories()
    logger.info("Directories initialized")
    
    # Pre-load the model
    _ = ai_model.model
    logger.info("AI Model loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier",
    description="A MobileNetV2-based image classifier for cats and dogs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include routers
app.include_router(api_router)
app.include_router(views_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
