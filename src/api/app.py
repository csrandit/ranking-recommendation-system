from fastapi import FastAPI, Depends, HTTPException
from typing import Tuple, Dict
import logging

from src.api.schemas import RecommendationResponse, HealthResponse
from src.api.dependencies import get_model_and_movies

# ===============================
# Logging Configuration
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("Starting Movie Recommendation API...")

# ===============================
# FastAPI App
# ===============================

app = FastAPI(
    title="🎬 Movie Recommendation API",
    version="1.1.0",
    description="Production-style Movie Recommendation System using Collaborative Filtering"
)

# ===============================
# Root Endpoint
# ===============================

@app.get("/", tags=["Root"])
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Movie Recommendation API is running 🚀"}

# ===============================
# Recommendation Endpoint
# ===============================

@app.get(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"]
)
def recommend(
    user_id: int,
    k: int = 10,
    resources: Tuple = Depends(get_model_and_movies)
):
    model, movie_dict = resources

    logger.info(f"Received recommendation request for user_id={user_id}, k={k}")

    try:
        recommendations = model.recommend(user_id=user_id, k=k)
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=404, detail="User not found")

    titles = [
        movie_dict.get(item_id, "Unknown Movie")
        for item_id in recommendations
    ]

    logger.info(f"Returning {len(titles)} recommendations")

    return RecommendationResponse(
        user_id=user_id,
        recommendations=titles
    )

# ===============================
# Health Check Endpoint
# ===============================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
def health_check():
    logger.info("Health check accessed")
    return HealthResponse(status="ok")