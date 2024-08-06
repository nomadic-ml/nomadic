from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
experiment = {
    experiment_runs = {
        experiment_run: {
            selected_hp_values = {
                'hp_name': HP_VALUE: Iterable
            },
        },
    },
    current_hp_values = {
        'hp_name': HP_VALUE: Iterable
    },
    hp_search_space_map = {
        'hp_name': hp_search_space
    },
    datetime_started = datetime,
    datetime_ended = datetime,
    author = User
}
"""


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
    """Base experiment run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required
    param_dict: Dict[str, Any] = Field(
        ..., description="A dictionary of parameters to iterate over."
    )
    evaluation_dataset: List[Dict] = Field(
        default_factory=List,
        description="Evaluation dataset in dictionary format.",
    )
    user_prompt_request: str = Field(
        default="",
        description="User request for GPT prompt.",
    )
    # TODO: Figure out why Union[SagemakerModel, OpenAIModel] doesn't work
    # Note: A model is always required. It is currently denoted as `Optional` because of the TODO above.
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluator: Optional[Union[BaseEvaluator, Callable]] = Field(
        default=None,
        description="Evaluator of experiment (BaseEvaluator instance or callable)",
    )
    # tuner is checked for being child of BaseParamTuner in `check_tuner_class`
    tuner: Optional[Any] = Field(default=None, description="Placeholder for base tuner")
    # Optional
    fixed_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of fixed hyperparameter values.",
    )
    current_param_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of current hyperparameter values.",
    )
    num_extra_prompts: Optional[int] = Field(
        default=0,
        description="Number of extra prompts to generate.",
    )
    num_example_prompts: Optional[int] = Field(
        default=0,
        description="Number of example prompts to include for few-shot prompting.",
    )
    search_method: Optional[str] = Field(
        default="grid", description="Tuner search option. Can be: [grid, bayesian]"
    )
    # Self populated
    start_datetime: Optional[datetime] = Field(
        default=None, description="Start datetime."
    )
    end_datetime: Optional[datetime] = Field(default=None, description="End datetime.")
    tuned_result: Optional[TunedResult] = Field(
        default=None, description="Tuned result of Experiment"
    )
    experiment_status: Optional[ExperimentStatus] = Field(
        default=ExperimentStatus("not_started"),
        description="Current status of Experiment",
    )
    experiment_status_message: Optional[str] = Field(
        default="",
        description="Detailed description of Experiment status during error.",
    )

    # Enhanced fields for multi-prompt approach
    prompting_approaches: List[str] = Field(
        default=["zero-shot"],
        description="List of prompting approaches to try (e.g., zero-shot, few-shot, chain-of-thought).",
    )
    prompt_complexities: List[str] = Field(
        default=["simple"],
        description="List of prompt complexity levels to try (e.g., simple, detailed, very detailed).",
    )
    prompt_focuses: List[str] = Field(
        default=[""],
        description="List of prompt focuses to try (e.g., fact extraction, action points, British English usage).",
    )
    num_iterations_per_prompt: int = Field(
        default=1,
        description="Number of times to run each prompt variant.",
    )

    # New fields for enhanced parameter handling
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate in the response.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Controls randomness in the output. Higher values mean more random completions.",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="How much to penalize new tokens based on their existing frequency in the text so far.",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="How much to penalize new tokens based on whether they appear in the text so far.",
    )

    # New fields for custom evaluator and evaluation metrics
    custom_evaluator: Optional[Callable[[str], dict]] = Field(
        default=None,
        description="Custom evaluator function that takes a response string and returns an evaluation prompt.",
    )
    evaluation_metrics: Optional[List[str]] = Field(
        default=None,
        description="List of evaluation metrics to be used with the custom evaluator.",
    )

    @field_validator("tuner")
    def check_tuner_class(cls, value):
        if not isinstance(value, BaseParamTuner):
            raise ValueError("tuner must be a subclass of BaseParamTuner")
        return value

    def model_post_init(self, ctx):
        if self.search_method not in ("grid", "bayesian"):
            raise ValueError(
                f"Selected Experiment search_method `{self.search_method}` is not valid."
            )

    def generate_similar_prompts(
        self, prompt: str, user_request: str, approach: str, complexity: str, focus: str
    ) -> List[str]:
        """
        Generate similar prompts using a GPT query.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.model.api_keys["OPENAI_API_KEY"])

        system_message = f"""
        Generate {self.num_iterations_per_prompt} prompt variants based on the following parameters:
        - Prompting Approach: {approach}
        - Prompt Complexity: {complexity}
        - Prompt Focus: {focus}

        Each prompt should be a variation of the original prompt, adhering to the specified parameters.
        """

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
            n=self.num_iterations_per_prompt,
            stop=None,
            temperature=0.7,
        )

        prompt_variants = [
            choice.message.content.strip() for choice in response.choices
        ]
        return prompt_variants

    def get_fewshot_prompt_template(self) -> FewShotPromptTemplate:
        """
        Generate few shot prompt template using LangChain
        """
        examples = self.evaluation_dataset
        # create a example template
        example_template = "Context: {Context}\n\nInstruction: {Instruction}\n\nQuestion: {Question}\n\nAnswer: {Answer}"

        # create a prompt example from above template
        example_prompt = PromptTemplate(
            input_variables=["Context", "Instruction", "Question", "Answer"],
            template=example_template,
        )

        # Choose between MaxMarginalRelevanceExampleSelector and SemanticSimilarityExampleSelector
        if self.prompting_approach == "mmr":
            example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                examples,
                OpenAIEmbeddings(),
                FAISS,
                k=self.num_example_prompts,
                fetch_k=len(examples),
            )
        else:
            example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples,
                OpenAIEmbeddings(),
                Chroma,
                k=self.num_example_prompts,
            )

        # Now create the few shot prompt template
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
            suffix="{input}\n\nAnswer:",
            example_separator="\n\n",
        )

        return few_shot_prompt

    # flake8: noqa: C901
    def run(self) -> TunedResult:
        """
        Run experiment.
        """
        is_error = False

        def get_responses(type_safe_param_values):
            all_pred_responses, all_eval_qs, all_ref_responses = [], [], []

            for approach in self.prompting_approaches:
                for complexity in self.prompt_complexities:
                    for focus in self.prompt_focuses:
                        print(
                            f"\nGenerating prompts for: Approach={approach}, Complexity={complexity}, Focus={focus}"
                        )
                        prompt_variants = self._generate_similar_prompts(
                            self.user_prompt_request,
                            self.user_prompt_request,
                            approach=approach,
                            complexity=complexity,
                            focus=focus,
                        )

                        for i, prompt_variant in enumerate(prompt_variants):
                            print(
                                f"\nProcessing prompt variant {i+1}/{len(prompt_variants)}"
                            )
                            print(
                                f"Prompt: {prompt_variant[:100]}..."
                            )  # Print first 100 chars of prompt
                            for iteration in range(self.num_iterations_per_prompt):
                                print(
                                    f"\nIteration {iteration+1}/{self.num_iterations_per_prompt}"
                                )
                                pred_responses, eval_qs, ref_responses = [], [], []
                                # If evaluation dataset is not provided
                                if not self.evaluation_dataset:
                                    completion_response: CompletionResponse = (
                                        self.model.run(
                                            prompt=prompt_variant,
                                            parameters=type_safe_param_values,
                                        )
                                    )
                                    pred_response = self._extract_response(
                                        completion_response
                                    )
                                    pred_responses.append(pred_response)
                                    eval_qs.append(prompt_variant)
                                    ref_responses.append(None)
                                    print(
                                        f"Response: {pred_response[:100]}..."
                                    )  # Print first 100 chars of response
                                # If evaluation dataset is provided
                                else:
                                    for j, example in enumerate(
                                        self.evaluation_dataset
                                    ):
                                        print(
                                            f"Processing example {j+1}/{len(self.evaluation_dataset)}"
                                        )
                                        prompt = self._construct_prompt(
                                            prompt_variant, example
                                        )
                                        completion_response: CompletionResponse = (
                                            self.model.run(
                                                prompt=prompt,
                                                parameters=type_safe_param_values,
                                            )
                                        )
                                        pred_response = self._extract_response(
                                            completion_response
                                        )
                                        pred_responses.append(pred_response)
                                        eval_qs.append(prompt)
                                        ref_responses.append(
                                            example.get("Answer", None)
                                        )
                                        print(
                                            f"Response: {pred_response[:100]}..."
                                        )  # Print first 100 chars of response
                                all_pred_responses.extend(pred_responses)
                                all_eval_qs.extend(eval_qs)
                                all_ref_responses.extend(ref_responses)
            return (all_pred_responses, all_eval_qs, all_ref_responses)

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            print("\nStarting new experiment run with parameters:")
            for param, value in param_values.items():
                print(f"{param}: {value}")

            type_safe_param_values = self._enforce_param_types(param_values)
            pred_responses, eval_qs, ref_responses = get_responses(
                type_safe_param_values
            )
            eval_results = self._evaluate_responses(pred_responses, ref_responses)
            mean_score = self._calculate_mean_score(eval_results)

            print(f"\nExperiment run completed. Mean score: {mean_score}")

            return RunResult(
                score=mean_score,
                params=param_values,
                metadata={
                    "Prompts": eval_qs,
                    "Answers": pred_responses,
                    "Ground Truth": ref_responses,
                    "Prompting Approaches": self.prompting_approaches,
                    "Prompt Complexities": self.prompt_complexities,
                    "Prompt Focuses": self.prompt_focuses,
                    "Evaluation Metrics": self.evaluation_metrics,
                    "Custom Evaluator Results": eval_results,
                },
            )

        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        result = None
        try:
            print("\nSetting up tuner...")
            self._setup_tuner(default_param_function)
            print("Starting experiment...")
            result = self.tuner.fit()
        except Exception as e:
            is_error = True
            self.experiment_status_message = self._format_error_message(e)
            print(f"Error occurred: {self.experiment_status_message}")

        self.end_datetime = datetime.now()
        self.experiment_status = self._determine_experiment_status(is_error)
        self.tuned_result = result or self._create_default_tuned_result()
        print(f"\nExperiment completed. Status: {self.experiment_status}")
        return self.tuned_result

    def _construct_prompt(self, prompt_variant: str, example: Dict[str, str]) -> str:
        prompt = f"{prompt_variant}\n\nContext: {example['Context']}\n\nInstruction: {example['Instruction']}\n\nQuestion: {example['Question']}\n\n"
        if self.num_example_prompts > 0:
            prompt = self.get_fewshot_prompt_template().format(input=prompt)
        return prompt

    def _extract_response(self, completion_response: CompletionResponse) -> str:
        if isinstance(self.model, OpenAIModel):
            return completion_response.text
        elif isinstance(self.model, SagemakerModel):
            return completion_response.raw["Body"]
        else:
            raise NotImplementedError("Unsupported model type")

    def _enforce_param_types(self, param_values: Dict[str, Any]) -> Dict[str, Any]:
        type_safe_param_values = {}
        for param, val in param_values.items():
            if param in self.model.hyperparameters.default:
                type_safe_param_values[param] = self.model.hyperparameters.default[
                    param
                ]["type"](val)
            else:
                type_safe_param_values[param] = val
        return type_safe_param_values

    def _evaluate_responses(
        self, pred_responses: List[str], ref_responses: List[str]
    ) -> List[Any]:
        eval_results = []
        if self.custom_evaluator:
            for pred in pred_responses:
                eval_results.append(
                    self.custom_evaluator(pred, self.evaluation_metrics)
                )
        elif self.evaluator:
            for pred, ref in zip(pred_responses, ref_responses):
                eval_results.append(
                    self.evaluator.evaluate_response(
                        response=Response(pred), reference=ref
                    )
                )
        return eval_results

    def _calculate_mean_score(self, eval_results: List[Any]) -> float:
        scores = [r.score for r in eval_results]
        return sum(scores) / len(scores) if scores else 0

    def _setup_tuner(self, param_function: Callable):
        if not self.tuner:
            if is_ray_installed():
                from nomadic.tuner.ray import RayTuneParamTuner

                self.tuner = RayTuneParamTuner(
                    param_fn=param_function,
                    param_dict=self.param_dict,
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=True,
                )
            else:
                from nomadic.tuner import FlamlParamTuner

                self.tuner = FlamlParamTuner(
                    param_fn=param_function,
                    param_dict=self.param_dict,
                    search_method=self.search_method,
                    fixed_param_dict=self.fixed_param_dict,
                    current_param_dict=self.current_param_dict,
                    show_progress=True,
                    num_samples=-1,
                )

    def _format_error_message(self, exception: Exception) -> str:
        return f"Exception: {str(exception)}\n\nStack Trace:\n{traceback.format_exc()}"

    def _determine_experiment_status(self, is_error: bool) -> ExperimentStatus:
        return (
            ExperimentStatus("finished_success")
            if not is_error
            else ExperimentStatus("finished_error")
        )

    def _create_default_tuned_result(self) -> TunedResult:
        return TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )

    def save_experiment(self, folder_path: Path):
        file_name = (
            f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))

    def visualize_results(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.tuned_result:
            print("No results to visualize.")
            return

        # Extract scores and parameters from run results
        scores = [run.score for run in self.tuned_result.run_results]
        params = [run.params for run in self.tuned_result.run_results]

        # Create a DataFrame for easier plotting
        import pandas as pd

        df = pd.DataFrame(params)
        df["score"] = scores

        # Set up the plot
        plt.figure(figsize=(12, 6))

        # Plot parameter distributions
        for i, param in enumerate(df.columns[:-1]):  # Exclude 'score'
            plt.subplot(2, 3, i + 1)
            sns.histplot(df[param], kde=True)
            plt.title(f"Distribution of {param}")

        # Plot score distribution
        plt.subplot(2, 3, 6)
        sns.histplot(df["score"], kde=True)
        plt.title("Distribution of Scores")

        plt.tight_layout()
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap of Parameters and Score")
        plt.show()

        # Scatter plots for top 2 most correlated parameters with score
        correlations = df.corr()["score"].abs().sort_values(ascending=False)
        top_params = correlations.index[1:3]  # Exclude score itself

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i, param in enumerate(top_params):
            sns.scatterplot(data=df, x=param, y="score", ax=axes[i])
            axes[i].set_title(f"{param} vs Score")

        plt.tight_layout()
        plt.show()

    def _generate_similar_prompts(
        self, prompt: str, user_request: str, approach: str, complexity: str, focus: str
    ) -> List[str]:
        """Generate similar prompts using the model."""
        return self.generate_similar_prompts(
            prompt, user_request, approach, complexity, focus
        )

    def custom_evaluate(self, response: str, evaluation_metrics: List[str]) -> dict:
        """
        Custom evaluation function for the experiment.

        Args:
            response (str): The response to evaluate.
            evaluation_metrics (List[str]): List of evaluation metrics to use.

        Returns:
            dict: A dictionary containing the evaluation prompt.
        """
        metrics_prompt = "\n".join(
            [f"{i+1}. {metric}" for i, metric in enumerate(evaluation_metrics)]
        )
        metrics_format = "\n".join(
            [f"{metric}: [score]" for metric in evaluation_metrics]
        )

        evaluation_prompt = f"""
        You are a judge evaluating the quality of a generated response for a financial advice summary.
        Please evaluate the following response based on these criteria, scoring each from 0 to 20:

        {metrics_prompt}

        Response to evaluate:
        {response}

        Provide your evaluation in the following format:
        {metrics_format}
        Overall score: [total out of {len(evaluation_metrics) * 20}]

        Brief explanation: [Your explanation of the strengths and weaknesses]
        """

        return {"evaluation_prompt": evaluation_prompt}
