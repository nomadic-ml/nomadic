"""Base experiment implementation supporting both parameter-based and dataset-based experiments."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Set
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import os

from nomadic.experiment.helpers import (
    BaseEvaluator as Evaluator,
    BaseResultManager as ResultManager,
    BaseExperimentSetup as ExperimentSetup,
    BaseResponseManager as ResponseManager,
    BasePromptConstructor as DefaultPromptConstructor
)

class BaseExperiment(ABC, BaseModel):
    """Abstract base class for all experiment types.

    Supports both parameter-based experiments (using param_fn) and dataset-based
    experiments (using evaluation_dataset and model). Only requires minimal core
    configuration with optional extended functionality.
    """

    # Required core configuration
    name: str = "default_experiment"
    params: Set[str]
    enable_logging: bool = False

    # Optional parameter-based experiment config
    param_fn: Optional[Callable] = None
    fixed_param_dict: Dict[str, Any] = Field(default_factory=dict)

    # Optional dataset-based experiment config
    evaluation_dataset: Optional[List[Dict[str, str]]] = None
    user_prompt_request: Optional[List[str]] = None
    model: Optional[Any] = None  # Can be any model class that supports generation
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    evaluator: Optional[Dict[str, str]] = None  # Evaluator config for dataset experiments

    # Optional experiment settings
    search_method: str = "grid"
    use_flaml_library: bool = False
    use_iterative_optimization: bool = False
    num_simulations: int = 1

    # Internal state
    start_datetime: datetime = Field(default_factory=datetime.now)

    @abstractmethod
    def run(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the experiment with the given parameters.

        This method must be implemented by concrete experiment classes.
        The implementation should handle experiment execution and return results.
        """
        pass

    def _validate_and_setup(self, param_dict: Dict[str, Any]) -> None:
        """Validate parameters and setup experiment."""
        if self.enable_logging:
            ExperimentSetup.setup_logging(self.name)

        if not ExperimentSetup.validate_params(param_dict, self.params):
            missing_params = self.params - set(param_dict.keys())
            raise ValueError(f"Missing required parameters: {missing_params}")

    def _run_dataset_experiment(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run experiment using evaluation dataset."""
        if not self.evaluation_dataset:
            raise ValueError("evaluation_dataset must be provided for dataset experiments")

        results = {}
        prompt_constructor = DefaultPromptConstructor()

        for item in self.evaluation_dataset:
            # Construct prompt using template and user prompt request if available
            prompt_params = {**param_dict, **item}
            if self.user_prompt_request:
                prompt_params["template"] = self.user_prompt_request[0]  # Use first template for now

            prompt = prompt_constructor.construct_prompt(
                params=prompt_params,
                examples=self.fixed_param_dict.get("examples")
            )

            # Get model response
            if self.model:
                response = self.model.generate(prompt, **self.model_kwargs)
            else:
                response = self.param_fn(prompt, **param_dict)

            # Validate and process response
            if ResponseManager.validate_response(response):
                structured_response = prompt_constructor.extract_response(response)

                # Evaluate response using configured evaluator
                score = None
                if "answer" in item and self.evaluator:
                    score = Evaluator.evaluate_response(
                        structured_response,
                        item["answer"],
                        method=self.evaluator.get("method", "accuracy_evaluator")
                    )

                results[item["query"]] = {
                    "response": structured_response,
                    "score": score,
                    "prompt": prompt  # Include prompt for analysis
                }

        return results

    def _run_parameter_experiment(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run traditional parameter-based experiment."""
        if not self.param_fn:
            raise ValueError("param_fn must be provided for parameter-based experiments")

        # Combine fixed and variable parameters
        full_params = {**self.fixed_param_dict, **param_dict}
        return self.param_fn(**full_params)

    def save_experiment(self, folder_path: Path):
        """Save experiment results."""
        file_name = f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(folder_path / file_name, "w") as file:
            file.write(self.model_dump_json())
