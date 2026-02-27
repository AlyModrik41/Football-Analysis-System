from pydantic import BaseModel

class MatchStats(BaseModel):
    team_1_possession: float
    team_2_possession: float
    neutral_possession: float
    total_passes: int