"""
This module defines the Experiment class and related components for running
machine learning experiments with various models and hyperparameter tuning.

It includes functionality for generating prompts, evaluating responses,
and managing experiment workflows.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.llms import CompletionResponse
from llama_index.core.base.response.schema import Response

from nomadic.model import OpenAIModel, SagemakerModel
from nomadic.result import RunResult, TunedResult
from nomadic.tuner.base import BaseParamTuner
from nomadic.util import is_ray_installed

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_openai import OpenAIEmbeddings


class ExperimentStatus(Enum):
    not_started = "not_started"
    running = "running"
    finished_success = "finished_success"
    finished_error = "finished_error"


class ExperimentMode(Enum):
    train = "training"
    fine_tune = "fine_tuning"
    inference = "inference"


class Experiment(BaseModel):
    """
    Represents an experiment run for model evaluation and hyperparameter tuning.

    This class encapsulates all the necessary components and settings for running
    an experiment, including model parameters, evaluation datasets, and tuning options.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required fields
    param_dict: Dict[str, Any] = Field(
        ..., description="Dictionary of parameters to iterate over during tuning."
    )
    user_prompt_request: str = Field(
        default="",
        description="User-provided request for generating the GPT prompt.",
    )
    # TODO: Implement Union[SagemakerModel, OpenAIModel] for proper typing
    model: Optional[Any] = Field(
        default=None,
        description="Model instance to run the experiment (SagemakerModel or OpenAIModel).",
    )
    # TODO: Figure out why Union[SagemakerModel, OpenAIModel] doesn't work
    # Note: A model is always required. It is currently denoted as `Optional` brcause of the TODO above.
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluation_dataset: Optional[List[Dict]] = Field(
        default=[{}],
        description="Evaluation dataset in dictionary format.",
    )
    evaluator: Optional[BaseEvaluator] = Field(
        default=None, description="Evaluator instance for assessing experiment results."
    )
    tuner: Optional[Any] = Field(
        default=None, description="Tuner instance for hyperparameter optimization."
    )

    # Optional fields
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of fixed hyperparameter values.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of current hyperparameter values.",
    )
    num_extra_prompts: int = Field(
        default=0,
        description="Number of additional prompts to generate.",
    )
    num_example_prompts: int = Field(
        default=0,
        description="Number of example prompts for few-shot learning.",
    )
    search_method: Optional[str] = Field(
        default="grid", description="Tuner search option. Can be: [grid, bayesian]"
    )

    # Experiment status fields
    start_datetime: Optional[datetime] = Field(
        default=None, description="Experiment start timestamp."
    )
    search_method: Optional[str] = Field(
        default="grid", description="Tuner search option. Can be: [grid, bayesian]"
    )
    results_filepath: Optional[str] = Field(
        default=None, description="Path of outputting tuner run results."
    )
    end_datetime: Optional[datetime] = Field(
        default=None, description="Experiment end timestamp."
    )
    tuned_result: Optional[TunedResult] = Field(
        default=None, description="Results of the tuning process."
    )
    experiment_status: ExperimentStatus = Field(
        default=ExperimentStatus.not_started,
        description="Current status of the experiment.",
    )
    experiment_status_message: str = Field(
        default="",
        description="Detailed status message, especially useful for error reporting.",
    )

    # Multi-prompt approach fields
    num_prompt_variants: int = Field(
        default=1,
        description="Number of prompt variants to generate and evaluate.",
    )
    prompt_variations: List[str] = Field(
        default_factory=list,
        description="List of generated prompt variations.",
    )
    num_iterations_per_prompt: int = Field(
        default=1,
        description="Number of iterations to run for each prompt variant.",
    )
    prompting_approach: str = Field(
        default="zero-shot",
        description="Prompting strategy (e.g., zero-shot, few-shot, chain-of-thought).",
    )
    prompt_complexity: str = Field(
        default="simple",
        description="Complexity level of the prompt (e.g., simple, detailed, very detailed).",
    )
    prompt_focus: str = Field(
        default="",
        description="Specific focus or emphasis for the prompt generation.",
    )

    @field_validator("tuner")
    def check_tuner_class(cls, value):
        """Ensure that the tuner is an instance of BaseParamTuner."""
        if value is not None and not isinstance(value, BaseParamTuner):
            raise ValueError("tuner must be a subclass of BaseParamTuner")
        return value

    @field_validator("search_method")
    def validate_search_method(cls, value):
        """Validate the search method."""
        valid_methods = ["grid", "bayesian"]
        if value not in valid_methods:
            raise ValueError(f"search_method must be one of {valid_methods}")
        return value

    def model_post_init(self, ctx):
        if self.search_method not in ("grid", "bayesian"):
            raise ValueError(
                f"Selected Experiment search_method `{self.search_method}` is not valid."
            )

    def generate_similar_prompts(self, prompt: str, user_request: str) -> List[str]:
        """
        Generate similar prompts using a GPT query.

        Args:
            prompt (str): The original prompt to base variations on.
            user_request (str): The user's specific request for prompt generation.

        Returns:
            List[str]: A list of generated prompt variants.

        Raises:
            OpenAIError: If there's an issue with the OpenAI API call.
        """
        from openai import OpenAI

        # Initialize OpenAI client with API key
        client = OpenAI(api_key=self.model.api_keys["OPENAI_API_KEY"])

        # Construct system message with experiment parameters
        system_message = f"""
        Generate {self.num_prompt_variants} prompt variants based on the following parameters:
        - Prompting Approach: {self.prompting_approach}
        - Prompt Complexity: {self.prompt_complexity}
        - Prompt Focus: {self.prompt_focus}

        Each prompt should be a variation of the original prompt, adhering to the specified parameters.
        """

        try:
            # Make API call to generate prompt variants
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Original prompt: {prompt}\nUser request: {user_request}",
                    },
                ],
                max_tokens=200,
                n=self.num_prompt_variants,
                stop=None,
                temperature=0.7,
            )

            # Extract and clean prompt variants from the response
            prompt_variants = [
                choice.message.content.strip() for choice in response.choices
            ]
            return prompt_variants

        except Exception as e:
            # Log the error and re-raise it
            print(f"Error generating similar prompts: {str(e)}")
            raise

    def get_fewshot_prompt_template(self) -> FewShotPromptTemplate:
        """
        Generate a few-shot prompt template using LangChain.

        This method creates a FewShotPromptTemplate that selects relevant examples
        from the evaluation dataset based on semantic similarity. It uses the
        SemanticSimilarityExampleSelector to choose the most appropriate examples
        for the given input.

        Returns:
            FewShotPromptTemplate: A template for few-shot learning prompts.
        """
        # Define the structure for each example in the prompt
        example_template = (
            "Context: {Context}\n\n"
            "Instruction: {Instruction}\n\n"
            "Question: {Question}\n\n"
            "Answer: {Answer}"
        )

        # Create a prompt template from the example structure
        example_prompt = PromptTemplate(
            input_variables=["Context", "Instruction", "Question", "Answer"],
            template=example_template,
        )

        # Initialize the SemanticSimilarityExampleSelector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=self.evaluation_dataset,  # List of examples to select from
            embeddings=OpenAIEmbeddings(),  # Embedding model for semantic similarity
            vectorstore_cls=Chroma,  # Vector store for similarity search
            k=self.num_example_prompts,  # Number of examples to include in each prompt
        )

        # Create and return the few-shot prompt template
        return FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
            suffix="{input}\n\nAnswer:",
            example_separator="\n\n",
        )

    def run(self) -> TunedResult:
        """
        Run the experiment and return the tuned result.

        This method orchestrates the entire experiment process, including:
        1. Generating prompt variants
        2. Running the model with different parameters
        3. Evaluating the responses
        4. Tuning the parameters

        Returns:
            TunedResult: The result of the tuning process, including the best parameters and scores.
        """
        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        result = None

        try:
            self._initialize_tuner()
            result = self.tuner.fit()
        except Exception as e:
            self._handle_experiment_error(e)

        self._finalize_experiment(result)
        return self.tuned_result

    def _initialize_tuner(self):
        """Initialize the tuner if not already set."""
        if not self.tuner:
            self.tuner = self._create_default_tuner()
        else:
            # Ensure fields given to Experiment are carried to tuner,
            # if tuner was provided.
            if self.param_dict:
                self.tuner.param_dict = self.param_dict
            if self.fixed_param_dict:
                self.tuner.fixed_param_dict = self.fixed_param_dict
            if self.results_filepath:
                self.tuner.results_filepath = self.results_filepath

    def _create_default_tuner(self):
        """Create a default tuner based on available libraries."""
        if is_ray_installed():
            from nomadic.tuner.ray import RayTuneParamTuner

            return RayTuneParamTuner(
                param_fn=self._default_param_function,
                param_dict=self.param_dict,
                search_method=self.search_method,
                fixed_param_dict=self.fixed_param_dict,
                current_param_dict=self.current_param_dict,
                show_progress=True,
            )
        else:
            from nomadic.tuner import FlamlParamTuner

            return FlamlParamTuner(
                param_fn=self._default_param_function,
                param_dict=self.param_dict,
                search_method=self.search_method,
                fixed_param_dict=self.fixed_param_dict,
                current_param_dict=self.current_param_dict,
                show_progress=True,
                num_samples=-1,
            )

    def _handle_experiment_error(self, error: Exception):
        """Handle and log any errors that occur during the experiment."""
        self.experiment_status = ExperimentStatus("finished_error")
        self.experiment_status_message = (
            f"Exception: {str(error)}\n\nStack Trace:\n{traceback.format_exc()}"
        )

    def _finalize_experiment(self, result: Optional[TunedResult]):
        """Finalize the experiment by setting end time and status."""
        self.end_datetime = datetime.now()
        self.experiment_status = (
            ExperimentStatus("finished_success")
            if result
            else ExperimentStatus("finished_error")
        )
        self.tuned_result = result or TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )

    def _default_param_function(self, param_values: Dict[str, Any]) -> RunResult:
        """
        Default parameter function for tuning.

        Args:
            param_values (Dict[str, Any]): The parameter values to test.

        Returns:
            RunResult: The result of running the model with the given parameters.
        """
        type_safe_param_values = self._get_type_safe_param_values(param_values)
        pred_responses, eval_qs, ref_responses = self._get_responses(
            type_safe_param_values
        )
        eval_results = self._evaluate_responses(pred_responses, ref_responses)
        mean_score = self._calculate_mean_score(eval_results)

        return RunResult(
            score=mean_score,
            params=param_values,
            metadata={
                "Prompts": eval_qs,
                "Answers": pred_responses,
                "Ground Truth": ref_responses,
            },
        )

    def _get_type_safe_param_values(
        self, param_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert parameter values to their default types if they exist."""
        return {
            param: (
                self.model.hyperparameters.default[param]["type"](val)
                if param in self.model.hyperparameters.default
                else val
            )
            for param, val in param_values.items()
        }

    def _get_responses(
        self, type_safe_param_values: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Get model responses for all prompt variants and iterations."""
        all_pred_responses, all_eval_qs, all_ref_responses = [], [], []
        prompt_variants = self.generate_similar_prompts(
            self.user_prompt_request, self.user_prompt_request
        )

        for prompt_variant in prompt_variants:
            for _ in range(self.num_iterations_per_prompt):
                pred_responses, eval_qs, ref_responses = self._process_prompt(
                    prompt_variant, type_safe_param_values
                )
                all_pred_responses.extend(pred_responses)
                all_eval_qs.extend(eval_qs)
                all_ref_responses.extend(ref_responses)

        return all_pred_responses, all_eval_qs, all_ref_responses

    def _process_prompt(
        self, prompt_variant: str, type_safe_param_values: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Process a single prompt variant."""
        if not self.evaluation_dataset:
            return self._process_single_prompt(prompt_variant, type_safe_param_values)
        return self._process_evaluation_dataset(prompt_variant, type_safe_param_values)

    def _process_single_prompt(
        self, prompt: str, type_safe_param_values: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Process a single prompt without an evaluation dataset."""
        completion_response = self.model.run(
            prompt=prompt, parameters=type_safe_param_values
        )
        return [self._get_model_response(completion_response)], [prompt], [None]

    def _process_evaluation_dataset(
        self, prompt_variant: str, type_safe_param_values: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Process the evaluation dataset for a prompt variant."""
        pred_responses, eval_qs, ref_responses = [], [], []
        for example in self.evaluation_dataset:
            prompt = self._create_prompt(prompt_variant, example)
            completion_response = self.model.run(
                prompt=prompt, parameters=type_safe_param_values
            )
            pred_responses.append(self._get_model_response(completion_response))
            eval_qs.append(prompt)
            ref_responses.append(example.get("Answer", None))
        return pred_responses, eval_qs, ref_responses

    def _create_prompt(self, prompt_variant: str, example: Dict[str, str]) -> str:
        """Create a prompt from a variant and an example."""
        prompt = f"{prompt_variant}\n\nContext: {example['Context']}\n\nInstruction: {example['Instruction']}"
        if self.num_example_prompts > 0:
            prompt = self.get_fewshot_prompt_template().format(input=prompt)
        return prompt

    def _get_model_response(self, completion_response: CompletionResponse) -> str:
        """Extract the response from the model's completion response."""
        if isinstance(self.model, OpenAIModel):
            return completion_response.text
        elif isinstance(self.model, SagemakerModel):
            return completion_response.raw["Body"]
        else:
            raise NotImplementedError("Unsupported model type")

    def _evaluate_responses(
        self, pred_responses: List[str], ref_responses: List[str]
    ) -> List[Any]:
        """Evaluate the model's responses if an evaluator is available."""
        if not self.evaluator:
            return []
        return [
            self.evaluator.evaluate_response(response=Response(pred), reference=ref)
            for pred, ref in zip(pred_responses, ref_responses)
        ]

    def _calculate_mean_score(self, eval_results: List[Any]) -> float:
        """Calculate the mean score from evaluation results."""
        scores = [r.score for r in eval_results]
        return sum(scores) / len(scores) if scores else 0

    def save_experiment(self, folder_path: Path) -> None:
        """
        Save the experiment details to a JSON file.

        Args:
            folder_path (Path): The directory path where the file will be saved.

        Raises:
            IOError: If there's an error writing the file.
        """
        try:
            file_name = (
                f"experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            file_path = folder_path / file_name
            with open(file_path, "w") as file:
                json_data = self.model_dump_json(exclude={"model", "evaluator"})
                file.write(json_data)
        except IOError as e:
            raise IOError(f"Error saving experiment to {file_path}: {str(e)}")
