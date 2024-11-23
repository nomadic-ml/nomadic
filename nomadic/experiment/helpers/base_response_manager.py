from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

def get_responses(
    self,
    type_safe_param_values: Tuple[Dict[str, Any], Dict[str, Any]],
    model: Any,
    prompt_constructor: Any,
    evaluation_dataset: Optional[List[Dict[str, Any]]] = None,
    user_prompt_request: Optional[str] = None,
    num_simulations: int = 1,
    enable_logging: bool = False,
    use_iterative_optimization: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Default implementation of response generation."""

    openai_params, prompt_tuning_params = type_safe_param_values
    pred_responses = []
    eval_questions = []
    ref_responses = []
    prompt_variants = []

    if enable_logging:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    try:
        # Handle evaluation dataset if provided
        if evaluation_dataset:
            for item in evaluation_dataset:
                question = item.get('question', '')
                reference = item.get('reference', '')

                prompt = prompt_constructor.construct_prompt(
                    template=prompt_tuning_params.get('template', ''),
                    params={'question': question, **prompt_tuning_params}
                )

                response = _get_model_response(model, prompt, openai_params)

                pred_responses.append(response)
                eval_questions.append(question)
                ref_responses.append(reference)
                prompt_variants.append(prompt)

                if enable_logging:
                    logger.info(f"Generated response for question: {question}")

        # Handle user prompt request if provided
        elif user_prompt_request:
            for _ in range(num_simulations):
                prompt = prompt_constructor.construct_prompt(
                    template=prompt_tuning_params.get('template', ''),
                    params={'question': user_prompt_request, **prompt_tuning_params}
                )

                response = _get_model_response(model, prompt, openai_params)

                pred_responses.append(response)
                prompt_variants.append(prompt)

                if enable_logging:
                    logger.info(f"Generated response for user prompt")

                if use_iterative_optimization:
                    # Add delay between iterations
                    time.sleep(1)

        return pred_responses, eval_questions, ref_responses, prompt_variants

    except Exception as e:
        if enable_logging:
            logger.error(f"Error generating responses: {str(e)}")
        return [], [], [], []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_model_response(model: Any, prompt: str, params: Dict[str, Any]) -> str:
    """Get response from model with retry logic."""
    try:
        completion = model.run(
                    prompt=prompt,
                    parameters=params,
                )
        return extract_response(completion)
    except Exception as e:
        logging.error(f"Error getting model response: {str(e)}")
        raise

def extract_response(completion_response: Any) -> str:
    """Extract response from completion object."""
    try:
        if hasattr(completion_response, 'choices') and completion_response.choices:
            # Handle OpenAI-style response
            return completion_response.choices[0].text.strip()
        elif isinstance(completion_response, str):
            # Handle string response
            return completion_response.strip()
        elif hasattr(completion_response, 'response'):
            # Handle custom response object
            return completion_response.response.strip()
        else:
            # Try to convert to string as fallback
            return str(completion_response).strip()
    except Exception as e:
        logging.error(f"Error extracting response: {str(e)}")
        return ""
