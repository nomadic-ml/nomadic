import base64
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import pandas as pd

from nomadic.util import is_json_serializable


class RunResult(BaseModel):
    """Run result"""

    score: float
    params: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata"
    )
    visualization: Optional[bytes] = Field(
        default_factory=bytes, description="Visualization"
    )

    def get_json(self):
        params_to_serialize = self.params.copy()
        for key, value in params_to_serialize.items():
            if not is_json_serializable(value):
                params_to_serialize[key] = "Redacted for serialization"
        metadata_to_serialize = self.metadata.copy()
        for key, value in metadata_to_serialize.items():
            if not is_json_serializable(value):
                metadata_to_serialize[key] = "Redacted for serialization"
        # TODO: Fix LlamaIndex's `EvaluationResult` object not giving JSON with model_dump_json
        # that exists as a key to the 'Custom Evaluator Results' key of metadata
        return {
            "score": self.score,
            "params": params_to_serialize,
            "metadata": metadata_to_serialize,
        }


class ExperimentResult(BaseModel):
    hp_search_space: Dict[str, Any] = Field(
        default=dict(), description="Hyperparameter search space of ExperimentResult"
    )
    run_results: List[RunResult]
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Metadata")
    best_idx: Optional[int] = Field(
        default=0, description="Position of the best RunResult in run_results."
    )
    name: Optional[str] = Field(default=None, description="Name of ExperimentResult")
    client_id: Optional[str] = Field(
        default=None, description="ID of ExperimentResult in Workspace"
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
        """Export ExperimentResult to a DataFrame."""
        data = []
        for run_result in self.run_results:
            metadata = {}
            if include_metadata:
                metadata = {
                    f"run_result_metadata_{k}": v
                    for k, v in run_result.metadata.items()
                }
            row = {
                "score": run_result.score,
                **{f"param_{k}": v for k, v in run_result.params.items()},
                **metadata,
            }
            data.append(row)

        # Adding experiment result metadata as a row (if needed)
        if include_metadata and self.metadata:
            experiment_metadata = {
                f"experiment_result_metadata_{k}": v for k, v in self.metadata.items()
            }
            data.append(experiment_metadata)

        return pd.DataFrame(data)

def load_run_results_from_file(filepath: str) -> List[RunResult]:
    """
    Loads a list of dictionaries from a JSON file,
    converting each dictionary into a RunResult object.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    run_results = []
    for item in data:
        if "visualization" in item and item["visualization"] is not None:
            item["visualization"] = base64.b64decode(item["visualization"])

        run_results.append(RunResult(**item))

    return run_results
