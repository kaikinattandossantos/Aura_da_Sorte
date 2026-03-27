from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, List

class MatchAnalyzeRequest(BaseModel):
    ataques_perigosos: Dict[str, Any] = Field(default_factory=dict)
    escanteios: Dict[str, Any] = Field(default_factory=dict)
    x: float | str | None = None
    y: float | str | None = None
    location: list[Any] | None = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    results: list[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")
