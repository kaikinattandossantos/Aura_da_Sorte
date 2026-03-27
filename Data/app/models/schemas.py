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


# ---------------------------------------------------------------------------
# B365 event models — usados pelo endpoint /matches/dashboard/{match_id}
# ---------------------------------------------------------------------------

class B365League(BaseModel):
    id: str = ""
    name: str = "Unknown League"
    cc: str = ""
    model_config = ConfigDict(extra="allow")


class B365Team(BaseModel):
    id: str = ""
    name: str = ""
    image_id: str = ""
    cc: str = ""
    model_config = ConfigDict(extra="allow")


class B365Stats(BaseModel):
    attacks: list[str] = Field(default_factory=lambda: ["0", "0"])
    ball_safe: list[str] = Field(default_factory=lambda: ["0", "0"])
    corners: list[str] = Field(default_factory=lambda: ["0", "0"])
    corner_f: list[str] = Field(default_factory=lambda: ["0", "0"])
    corner_h: list[str] = Field(default_factory=lambda: ["0", "0"])
    dangerous_attacks: list[str] = Field(default_factory=lambda: ["0", "0"])
    goals: list[str] = Field(default_factory=lambda: ["0", "0"])
    off_target: list[str] = Field(default_factory=lambda: ["0", "0"])
    on_target: list[str] = Field(default_factory=lambda: ["0", "0"])
    penalties: list[str] = Field(default_factory=lambda: ["0", "0"])
    possession_rt: list[str] = Field(default_factory=lambda: ["50", "50"])
    redcards: list[str] = Field(default_factory=lambda: ["0", "0"])
    substitutions: list[str] = Field(default_factory=lambda: ["0", "0"])
    yellowcards: list[str] = Field(default_factory=lambda: ["0", "0"])
    yellowred_cards: list[str] = Field(default_factory=lambda: ["0", "0"])
    model_config = ConfigDict(extra="allow")


class B365Extra(BaseModel):
    length: int = 90
    home_pos: str = ""
    away_pos: str = ""
    numberofperiods: str = "2"
    periodlength: str = "45"
    round: str = ""
    model_config = ConfigDict(extra="allow")


class B365Event(BaseModel):
    id: str = ""
    text: str = ""
    model_config = ConfigDict(extra="allow")


class B365MatchResult(BaseModel):
    id: str = ""
    sport_id: str = ""
    time: str = ""
    time_status: str = ""
    league: B365League = Field(default_factory=B365League)
    home: B365Team = Field(default_factory=B365Team)
    away: B365Team = Field(default_factory=B365Team)
    ss: str = "0-0"
    scores: Dict[str, Any] = Field(default_factory=dict)
    stats: B365Stats = Field(default_factory=B365Stats)
    extra: B365Extra = Field(default_factory=B365Extra)
    events: list[B365Event] = Field(default_factory=list)
    has_lineup: int = 0
    inplay_created_at: str = ""
    inplay_updated_at: str = ""
    confirmed_at: str = ""
    bet365_id: str = ""
    model_config = ConfigDict(extra="allow")


class B365MatchEvent(BaseModel):
    success: int = 1
    results: list[B365MatchResult] = Field(default_factory=list)
