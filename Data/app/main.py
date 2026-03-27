from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.ml_manager import init_models
from app.api import matches, cards


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models and datasets on startup
    init_models()
    yield
    # Clean up on shutdown if necessary

app = FastAPI(
    title="Analytics API - Performance Pro",
    description="API de predições e análises ao vivo de futebol.",
    version="1.0.0",
    lifespan=lifespan
)

# Registra as rotas (Clean Architecture)
app.include_router(matches.router)
app.include_router(cards.router)

@app.get("/")
async def root():
    return {"status": "online", "api": "Analytics Performance"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
