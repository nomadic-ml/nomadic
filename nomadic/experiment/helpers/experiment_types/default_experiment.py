"""Default experiment implementation that supports both parameter-based and dataset-based experiments."""
from typing import Any, Dict, Optional
from datetime import datetime
import logging

from .base_experiment import BaseExperiment
from nomadic.experiment.helpers.base_evaluator import (
    accuracy_evaluator,
    transform_eval_dataset_to_eval_json,
    evaluate_responses,
    calculate_mean_score
)
from nomadic.experiment.helpers.base_result_manager import (
    format_error_message,
    determine_experiment_status,
    create_default_experiment_result,
    save_experiment
)
from nomadic.experiment.helpers.base_setup import setup_tuner, enforce_param_types
from nomadic.experiment.helpers.base_response_manager import get_responses
from nomadic.experiment.helpers.base_prompt_constructor import construct_prompt

class DefaultExperiment(BaseExperiment):
    """Default implementation of BaseExperiment supporting both parameter-based and dataset-based experiments.

    This implementation provides concrete implementations of all abstract methods
    and adds additional functionality for experiment management and evaluation.
    """

    def run(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the experiment with the given parameters.

        Args:
            param_dict: Dictionary containing experiment parameters

        Returns:
            Dictionary containing experiment results and metadata
        """
        try:
            # Setup experiment logging if enabled
            if self.enable_logging:
                logging.basicConfig(level=logging.INFO)
                logging.info(f"Starting experiment: {self.name}")
                logging.info(f"Parameters: {param_dict}")

            # Enforce parameter types and prepare parameters
            typed_params = enforce_param_types(param_dict, self.model_hyperparameters)
            openai_params, prompt_params = self._split_parameters(typed_params)

            # Setup tuner if using iterative optimization
            if self.use_iterative_optimization:
                tuning_params = setup_tuner(
                    self,
                    typed_params,
                    param_function=None,  # Can be extended to support custom parameter functions
                    experiment_name=self.name
                )
                prompt_params.update(tuning_params)

            # Get model responses
            pred_responses, eval_questions, ref_responses, prompts = get_responses(
                self,
                type_safe_param_values=(openai_params, prompt_params),
                model=self.model,
                prompt_constructor=self,
                evaluation_dataset=self.evaluation_dataset,
                user_prompt_request=self.user_prompt_request,
                num_simulations=self.num_simulations,
                enable_logging=self.enable_logging,
                use_iterative_optimization=self.use_iterative_optimization
            )

            # Transform evaluation dataset if provided
            if self.evaluation_dataset:
                eval_json = transform_eval_dataset_to_eval_json(self.evaluation_dataset)
            else:
                eval_json = None

            # Evaluate responses
            eval_results = evaluate_responses(
                self,
                responses=pred_responses,
                ref_responses=ref_responses,
                evaluation_dataset=eval_json,
                evaluator=accuracy_evaluator
            )

            # Calculate aggregate scores
            mean_scores = calculate_mean_score(
                self,
                eval_results=eval_results,
                params=self.params
            )

            # Prepare final results
            results = {
                "experiment_name": self.name,
                "status": determine_experiment_status(self, is_error=False),
                "parameters": typed_params,
                "predictions": pred_responses,
                "prompts": prompts,
                "evaluation": {
                    "results": eval_results,
                    "mean_scores": mean_scores,
                    "questions": eval_questions if eval_questions else None,
                    "references": ref_responses if ref_responses else None
                },
                "metadata": {
                    "timestamp": self.start_datetime.isoformat(),
                    "model": str(self.model.__class__.__name__) if self.model else None,
                    "num_simulations": self.num_simulations,
                    "optimization_enabled": self.use_iterative_optimization
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
                logging.info(f"Experiment completed successfully: {self.name}")

            return results

        except Exception as e:
            error_msg = format_error_message(self, e)
            if self.enable_logging:
                logging.error(error_msg)

            error_result = create_default_experiment_result(self, param_dict)
            error_result.update({
                "error": str(e),
                "error_trace": error_msg
            })

            return error_result

    def construct_prompt(self, template: str, params: Dict[str, Any], **kwargs) -> str:
        """Construct prompt using the prompt constructor helper."""
        return construct_prompt(self, template or self.prompt_template, params, **kwargs)

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the experiment."""
        return {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "template": self.prompt_template
        }
