from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from contextlib import asynccontextmanager

from app.api.upload import router as upload_router
from app.api.analysis import router as analysis_router
from app.api.mitigation import router as mitigation_router
from app.api.comprehensive_mitigation import router as comprehensive_mitigation_router
from app.api.ai_recommendations import router as ai_router
from app.utils.logger import get_api_logger

# Initialize logger
logger = get_api_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Fairness Assessment Platform starting up...")
    
    # Create upload directories if they don't exist
    os.makedirs("uploads/models", exist_ok=True)
    os.makedirs("uploads/datasets", exist_ok=True)
    os.makedirs("uploads/temp", exist_ok=True)
    logger.info("Created upload directories")
    
    yield
    
    # Shutdown
    logger.info("🛑 Fairness Assessment Platform shutting down...")


app = FastAPI(
    title="AI Fairness Assessment Platform",
    description="Comprehensive AI model fairness assessment with bias detection, mitigation, and AI-powered recommendations",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(mitigation_router, prefix="/api/mitigation", tags=["mitigation"])
app.include_router(comprehensive_mitigation_router, prefix="/api/mitigation", tags=["comprehensive_mitigation"])
app.include_router(ai_router, prefix="/api/ai", tags=["ai_recommendations"])

# Mount static files
app.mount("/static", StaticFiles(directory="uploads"), name="static")


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "AI Fairness Assessment Platform API",
        "version": "3.0.0",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy", "service": "fairness-assessment-api"}


if __name__ == "__main__":
    logger.info("Starting Fairness Assessment Platform server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
