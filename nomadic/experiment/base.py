from abc import ABC
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from pathlib import Path
import logging
import time

from nomadic.experiment.helpers.base_prompt_constructor import construct_prompt
from nomadic.experiment.helpers.base_response_manager import get_responses
from nomadic.experiment.helpers.base_evaluator import (
    custom_evaluate,
    custom_evaluate_hallucination,
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
from nomadic.experiment.helpers.experiment_types.base_experiment import BaseExperiment
from llama_index.core.evaluation import BaseEvaluator

class Experiment(BaseExperiment):
    """Experiment class that integrates all helper functions and implements BaseExperiment.

    This class provides a complete implementation that properly inherits from BaseExperiment
    and integrates all helper functions for prompt construction, response management,
    evaluation, and result handling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core configuration from base experiment
    name: str = Field(default="my experiment", description="Name of experiment")
    params: Optional[Set[str]] = Field(default_factory=set)
    enable_logging: bool = Field(default=True)

    # Experiment configuration
    evaluation_dataset: Optional[List[Dict[str, Any]]] = Field(default=None)
    user_prompt_request: Optional[Union[str, List[str]]] = Field(default_factory=list)
    model: Optional[Any] = Field(default=None)

    # Tuning and optimization settings
    use_iterative_optimization: Optional[Any] = Field(default=False)
    search_method: str = Field(default="grid")
    use_flaml_library: bool = Field(default=False)
    use_ray_backend: bool = Field(default=False)
    num_samples: int = Field(default=-1)

    # Evaluation configuration
    evaluator: Optional[Union[BaseEvaluator, Callable, Dict[str, Any]]] = Field(default=None)

    # Results and state management
    results_filepath: Optional[str] = Field(default=None)
    start_datetime: Optional[datetime] = Field(default=None)
    end_datetime: Optional[datetime] = Field(default=None)
    experiment_status: Optional[str] = Field(default="not_started")
    experiment_status_message: Optional[str] = Field(default="")

    def run(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the experiment with given parameters."""
        # Call parent class run method first
        result = super().run(param_dict)

        # Additional functionality specific to this implementation
        try:
            # Setup experiment logging
            if self.enable_logging:
                logging.basicConfig(level=logging.INFO)
                logging.info(f"Starting experiment: {self.name}")

            # Get responses using integrated response manager
            pred_responses, eval_questions, ref_responses, prompts = self._get_responses(
                self._enforce_param_types(param_dict)
            )

            # Evaluate responses using integrated evaluator
            eval_results = self._evaluate_responses(
                pred_responses,
                ref_responses,
                self.evaluation_dataset
            )

            # Calculate scores using integrated scoring
            mean_scores = self._calculate_mean_score(eval_results)

            # Prepare results using base result manager
            results = {
                "status": determine_experiment_status(self, is_error=False),
                "parameters": param_dict,
                "predictions": pred_responses,
                "prompts": prompts,
                "evaluation": {
                    "results": eval_results,
                    "mean_scores": mean_scores,
                    "questions": eval_questions,
                    "references": ref_responses
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": str(self.model.__class__.__name__) if self.model else None
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

    def _get_responses(
        self,
        type_safe_param_values: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get responses using integrated response manager."""
        return get_responses(
            self,
            type_safe_param_values=type_safe_param_values,
            model=self.model,
            prompt_constructor=self,
            evaluation_dataset=self.evaluation_dataset,
            user_prompt_request=self.user_prompt_request,
            num_simulations=self.num_simulations,
            enable_logging=self.enable_logging,
            use_iterative_optimization=self.use_iterative_optimization
        )

    def _evaluate_responses(
        self,
        pred_responses: List[str],
        ref_responses: Optional[List[str]] = None,
        evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate responses using integrated evaluator."""
        if self.evaluator:
            if isinstance(self.evaluator, dict):
                method = self.evaluator.get("method")
                if method == "custom_evaluate_hallucination":
                    return [
                        custom_evaluate_hallucination(pred, ref, evaluation_dataset)
                        for pred, ref in zip(pred_responses, ref_responses)
                    ]
                elif method == "custom_evaluate":
                    return [
                        custom_evaluate(
                            pred,
                            self.evaluator.get("evaluation_metrics", []),
                            self.model.api_keys.get("OPENAI_API_KEY") if hasattr(self.model, "api_keys") else None
                        )
                        for pred in pred_responses
                    ]
            else:
                return evaluate_responses(
                    self,
                    responses=pred_responses,
                    ref_responses=ref_responses,
                    evaluator=self.evaluator,
                    evaluation_dataset=evaluation_dataset
                )

        return evaluate_responses(
            self,
            responses=pred_responses,
            ref_responses=ref_responses,
            evaluation_dataset=evaluation_dataset
        )

    def _calculate_mean_score(self, eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate mean scores using integrated scoring."""
        return calculate_mean_score(self, eval_results=eval_results, params=self.params)

    def _enforce_param_types(self, param_values: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enforce parameter types using integrated type enforcement."""
        # First use the base setup helper
        typed_params = enforce_param_types(param_values, self.model.hyperparameters if hasattr(self.model, 'hyperparameters') else {})

        # Then split into OpenAI and prompt tuning parameters
        openai_params = {
            k: v for k, v in typed_params.items()
            if k in {'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty'}
        }

        prompt_params = {
            k: v for k, v in typed_params.items()
            if k not in openai_params
        }

        return openai_params, prompt_params

    def construct_prompt(self, template: str, params: Dict[str, Any], **kwargs) -> str:
        """Construct prompt using integrated prompt constructor."""
        return construct_prompt(self, template, params, **kwargs)
