from pydantic import BaseModel
from typing import List


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str