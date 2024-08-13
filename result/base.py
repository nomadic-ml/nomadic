from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import pandas as pd


class RunResult(BaseModel):
    """Run result"""

    score: float
    params: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata"
    )


class TunedResult(BaseModel):
    run_results: List[RunResult]
    best_idx: Optional[int] = Field(
        default=0, description="Position of the best RunResult in run_results."
    )

    def model_post_init(self, __context):
        self.run_results = sorted(self.run_results, key=lambda x: x.score, reverse=True)
        self.best_idx = 0

    @property
    def best_run_result(self) -> RunResult:
        """Get best run result."""
        return self.run_results[self.best_idx]

    def __len__(self) -> int:
        return len(self.run_results)

    def to_df(self, include_metadata=True) -> pd.DataFrame:
        """Export TunedResult to a DataFrame."""
        data = []
        for run_result in self.run_results:
            metadata = {}
            if include_metadata:
                metadata = {f"metadata_{k}": v for k, v in run_result.metadata.items()}
            row = {
                "score": run_result.score,
                **{f"param_{k}": v for k, v in run_result.params.items()},
                **metadata,
            }
            data.append(row)
        return pd.DataFrame(data)
