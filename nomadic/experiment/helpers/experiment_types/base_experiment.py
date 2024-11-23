from abc import ABC
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import logging
import time

from nomadic.experiment.helpers.base_prompt_constructor import construct_prompt
from nomadic.experiment.helpers.base_response_manager import get_responses
from nomadic.experiment.helpers.base_evaluator import (
    evaluate_responses,
    calculate_mean_score
)
from nomadic.experiment.helpers.base_result_manager import (
    format_error_message,
    determine_experiment_status,
    create_default_experiment_result,
    save_experiment
)
from nomadic.experiment.helpers.base_setup import enforce_param_types

class BaseExperiment(ABC, BaseModel):
    """Abstract base class for all experiment types.

    Supports both parameter-based experiments and dataset-based experiments with
    integrated evaluation, response management, and result handling.
    """

    # Required core configuration
    name: str = "default_experiment"
    params: Set[str] = Field(default_factory=set)
    enable_logging: bool = False

    # Experiment configuration
    evaluation_dataset: Optional[List[Dict[str, Any]]] = None
    user_prompt_request: Optional[str] = None
    model: Optional[Any] = None
    prompt_template: str = ""

    # Optional settings
    num_simulations: int = 1
    use_iterative_optimization: bool = False
    model_hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Internal state
    start_datetime: datetime = Field(default_factory=datetime.now)

    def run(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the experiment with given parameters."""
        try:
            # Setup and validation
            if self.enable_logging:
                logging.basicConfig(level=logging.INFO)

            # Enforce parameter types based on model hyperparameters
            typed_params = enforce_param_types(param_dict, self.model_hyperparameters)

            # Split parameters into OpenAI and prompt tuning params
            openai_params, prompt_params = self._split_parameters(typed_params)

            # Get responses using the response manager
            pred_responses, eval_questions, ref_responses, prompts = get_responses(
                self,
                type_safe_param_values=(openai_params, prompt_params),
                model=self.model,
                prompt_constructor=self,  # self implements construct_prompt
                evaluation_dataset=self.evaluation_dataset,
                user_prompt_request=self.user_prompt_request,
                num_simulations=self.num_simulations,
                enable_logging=self.enable_logging,
                use_iterative_optimization=self.use_iterative_optimization
            )

            # Evaluate responses
            eval_results = evaluate_responses(
                self,
                responses=pred_responses,
                ref_responses=ref_responses,
                evaluation_dataset=self.evaluation_dataset
            )

            # Calculate mean scores
            mean_scores = calculate_mean_score(
                self,
                eval_results=eval_results,
                params=self.params
            )

            # Prepare results
            results = {
                "status": determine_experiment_status(self, is_error=False),
                "parameters": typed_params,
                "predictions": pred_responses,
                "prompts": prompts,
                "evaluation": {
                    "results": eval_results,
                    "mean_scores": mean_scores
                },
                "metadata": {
                    "timestamp": self.start_datetime.isoformat(),
                    "model": str(self.model.__class__.__name__) if self.model else None,
                    "num_simulations": self.num_simulations
                }
            }

            # Save results if logging is enabled
            if self.enable_logging:
                save_experiment(
                    self,
                    folder_path=f"experiments/{self.name}",
                    start_datetime=self.start_datetime,
                    experiment_data=results
                )

            return results

        except Exception as e:
            error_msg = format_error_message(self, e)
            if self.enable_logging:
                logging.error(error_msg)

            return create_default_experiment_result(self, param_dict)

    def _split_parameters(self, param_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split parameters into OpenAI parameters and prompt tuning parameters."""
        openai_params = {
            k: v for k, v in param_dict.items()
            if k in {'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty'}
        }

        prompt_params = {
            k: v for k, v in param_dict.items()
            if k not in openai_params
        }

        return openai_params, prompt_params

    def construct_prompt(self, template: str, params: Dict[str, Any], **kwargs) -> str:
        """Construct prompt using the prompt constructor helper."""
        return construct_prompt(self, template or self.prompt_template, params, **kwargs)

    def extract_response(self, response: str, **kwargs) -> str:
        """Extract response content using the prompt constructor helper."""
        return construct_prompt(self).extract_response(response, **kwargs)

    class Config:
        arbitrary_types_allowed = True
