from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RunResult(BaseModel):
    """Run result."""

    score: float
    params: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata.")


class TunedResult(BaseModel):
    run_results: List[RunResult]
    best_idx: int

    @property
    def best_run_result(self) -> RunResult:
        """Get best run result."""
        return self.run_results[self.best_idx]
