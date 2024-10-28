from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel
import numpy as np
from statistics import mean, median

class BaseEvaluator(ABC, BaseModel):
    """Abstract base class for response evaluation."""

    @abstractmethod
    def evaluate_responses(self, responses: List[str], **kwargs) -> List[float]:
        """Evaluate a list of responses and return scores."""
        pass

    @abstractmethod
    def calculate_mean_score(
        self,
        eval_results: List[Dict[str, Any]],
        params: Optional[Set[str]] = None,
        evaluator: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate mean score from evaluation results.

        Args:
            eval_results: List of evaluation results containing scores and params
            params: Optional set of parameter names to group by
            evaluator: Optional custom evaluator instance
            **kwargs: Additional calculation parameters

        Returns:
            Dictionary mapping parameter combinations to mean scores
        """
        pass

    def handle_evaluation_error(self, error: Exception) -> float:
        """Handle evaluation errors with a default score."""
        print(f"Evaluation error: {str(error)}")
        return -1.0

class DefaultEvaluator(BaseEvaluator):
    """Default implementation of response evaluation."""

    def evaluate_responses(
        self,
        responses: List[str],
        ref_responses: Optional[List[str]] = None,
        evaluator: Optional[Any] = None,
        evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate responses against reference answers if provided, otherwise use heuristic scoring.

        Args:
            responses: List of response strings to evaluate
            ref_responses: Optional list of reference answers to compare against
            evaluator: Optional custom evaluator instance
            evaluation_dataset: Optional evaluation dataset with additional metadata
            **kwargs: Additional evaluation parameters

        Returns:
            List of evaluation results containing scores and metadata
        """
        try:
            eval_results = []
            for i, response in enumerate(responses):
                if not response:
                    eval_results.append({"score": 0.0})
                    continue

                result = {"response": response}

                if ref_responses and i < len(ref_responses):
                    # Compare with reference answer
                    ref = ref_responses[i]
                    score = self._calculate_similarity(response, ref)
                    result["reference"] = ref
                else:
                    # Heuristic scoring based on response length and structure
                    score = self._heuristic_score(response)

                result["score"] = min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

                # Add evaluation dataset metadata if available
                if evaluation_dataset and i < len(evaluation_dataset):
                    result.update(evaluation_dataset[i])

                eval_results.append(result)

            return eval_results

        except Exception as e:
            return [{"score": self.handle_evaluation_error(e)} for _ in responses]

    def calculate_mean_score(
        self,
        eval_results: List[Dict[str, Any]],
        params: Optional[Set[str]] = None,
        evaluator: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate mean score using specified method.

        Args:
            eval_results: List of evaluation results containing scores and params
            params: Optional set of parameter names to group by
            evaluator: Optional custom evaluator instance
            **kwargs: Additional calculation parameters

        Returns:
            Dictionary mapping parameter combinations to mean scores
        """
        if not eval_results:
            return {}

        try:
            # Use custom evaluator if provided and has calculate_mean_score method
            if evaluator and hasattr(evaluator, 'calculate_mean_score'):
                return evaluator.calculate_mean_score(eval_results)

            mean_scores = {}
            for result in eval_results:
                # Get params from the evaluation result or use provided params
                result_params = result.get("params", {})
                param_dict = {
                    k: result_params.get(k)
                    for k in (params or result_params.keys())
                }
                score = result.get("score", 0.0)

                # Create param key similar to original implementation
                param_key = str(tuple(sorted((k, param_dict[k]) for k in param_dict)))

                if param_key not in mean_scores:
                    mean_scores[param_key] = []
                mean_scores[param_key].append(score)

            # Calculate final mean scores for each param combination
            return {
                k: float(mean(v)) if v else 0.0
                for k, v in mean_scores.items()
            }

        except Exception as e:
            print(f"Error calculating mean score: {str(e)}")
            return {}

    def _calculate_similarity(self, response: str, reference: str) -> float:
        """Calculate similarity score between response and reference."""
        if not response or not reference:
            return 0.0

        # Simple word overlap similarity
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())

        if not reference_words:
            return 0.0

        overlap = len(response_words.intersection(reference_words))
        return overlap / len(reference_words)

    def _heuristic_score(self, response: str) -> float:
        """Calculate heuristic score based on response characteristics."""
        if not response:
            return 0.0

        # Basic scoring based on length and structure
        words = response.split()
        word_count = len(words)

        if word_count < 3:
            return 0.2
        elif word_count < 10:
            return 0.4
        elif word_count < 50:
            return 0.6
        else:
            return 0.8
