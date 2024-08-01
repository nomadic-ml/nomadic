from datetime import datetime
from enum import Enum
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional
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
    # Note: A model is always required. It is currently denoted as `Optional` brcause of the TODO above.
    model: Optional[Any] = Field(default=None, description="Model to run experiment")
    evaluator: Optional[BaseEvaluator] = Field(
        default=None, description="Evaluator of experiment"
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

    # New fields for multi-prompt approach
    num_prompt_variants: int = Field(
        default=1,
        description="Number of prompt variants to generate.",
    )
    num_iterations_per_prompt: int = Field(
        default=1,
        description="Number of times to run each prompt variant.",
    )
    prompting_approach: str = Field(
        default="zero-shot",
        description="Prompting approach (e.g., zero-shot, few-shot, chain-of-thought).",
    )
    prompt_complexity: str = Field(
        default="simple",
        description="Level of detail in the prompt (e.g., simple, detailed, very detailed).",
    )
    prompt_focus: str = Field(
        default="",
        description="Emphasis for the prompt (e.g., fact extraction, action points, British English usage).",
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

    def generate_similar_prompts(self, prompt: str, user_request: str) -> List[str]:
        """
        Generate similar prompts using a GPT query.
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.model.api_keys["OPENAI_API_KEY"])

        system_message = f"""
        Generate {self.num_prompt_variants} prompt variants based on the following parameters:
        - Prompting Approach: {self.prompting_approach}
        - Prompt Complexity: {self.prompt_complexity}
        - Prompt Focus: {self.prompt_focus}

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
            n=self.num_prompt_variants,
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

        """
        Choose between: MaxMarginalRelevanceExampleSelector OR SemanticSimilarityExampleSelector
        MMR selects examples based on a combination of
        which examples are most similar to the inputs, while also optimizing for diversity.
        It does this by finding the examples with the embeddings that have the greatest cosine
        similarity with the inputs, and then iteratively adding them while penalizing them
        for closeness to already selected examples.
        """
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # List of examples available to select from.
            examples,
            # Embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # VectorStore class that is used to store the embeddings and do a similarity search over.
            Chroma,  # Chroma for SemanticSimilarityExampleSelector, FAISS for MaxMarginalRelevanceExampleSelector
            # Number of examples to produce.
            k=self.num_example_prompts,
            # fetch_k=len(examples),  # Specify this IF using MMR,
        )
        # Now create the few shot prompt template
        mmr_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
            suffix="{input}\n\nAnswer:",
            example_separator="\n\n",
        )

        return mmr_prompt

    # flake8: noqa: C901
    def run(self) -> TunedResult:
        """
        Run experiment.
        """
        is_error = False

        def get_responses(type_safe_param_values):
            all_pred_responses, all_eval_qs, all_ref_responses = [], [], []

            prompt_variants = self.generate_similar_prompts(
                self.user_prompt_request, self.user_prompt_request
            )

            for prompt_variant in prompt_variants:
                for _ in range(self.num_iterations_per_prompt):
                    pred_responses, eval_qs, ref_responses = [], [], []
                    # If evaluation dataset is not provided
                    if not self.evaluation_dataset:
                        completion_response: CompletionResponse = self.model.run(
                            prompt=prompt_variant,
                            parameters=type_safe_param_values,
                        )
                    # If evaluation dataset is provided
                    else:
                        for example in self.evaluation_dataset:
                            prompt = f"{prompt_variant}\n\nContext: {example['Context']}\n\nInstruction: {example['Instruction']}\n\nQuestion: {example['Question']}\n\n"
                            if self.num_example_prompts > 0:
                                prompt = self.get_fewshot_prompt_template().format(
                                    input=prompt
                                )
                            completion_response: CompletionResponse = self.model.run(
                                prompt=prompt,
                                parameters=type_safe_param_values,
                            )
                            # OpenAI's model returns result in `completion_response.text`.
                            # Sagemaker's model returns result in `completion_response.raw["Body"]`.
                            if self.model:
                                if isinstance(self.model, OpenAIModel):
                                    pred_response = completion_response.text
                                elif isinstance(self.model, SagemakerModel):
                                    pred_response = completion_response.raw["Body"]
                                else:
                                    raise NotImplementedError
                                pred_responses.append(pred_response)
                                eval_qs.append(prompt)
                                ref_responses.append(example.get("Answer", None))
                    all_pred_responses.extend(pred_responses)
                    all_eval_qs.extend(eval_qs)
                    all_ref_responses.extend(ref_responses)
            return (all_pred_responses, all_eval_qs, all_ref_responses)

        def default_param_function(param_values: Dict[str, Any]) -> RunResult:
            # Enforce param values to fit their default types, if they exist.
            type_safe_param_values = {}
            for param, val in param_values.items():
                if param in self.model.hyperparameters.default:
                    type_safe_param_values[param] = self.model.hyperparameters.default[
                        param
                    ]["type"](val)
                else:
                    type_safe_param_values[param] = val
            pred_responses, eval_qs, ref_responses = get_responses(
                type_safe_param_values
            )
            eval_results = []
            if self.evaluator:
                for i, response in enumerate(pred_responses):
                    eval_results.append(
                        self.evaluator.evaluate_response(
                            response=Response(response), reference=ref_responses[i]
                        )
                    )

            # TODO: Generalize
            # get semantic similarity metric
            scores = [r.score for r in eval_results]
            mean_score = sum(scores) / len(scores) if scores else 0
            return RunResult(
                score=mean_score,
                params=param_values,
                metadata={
                    "Prompts": eval_qs,
                    "Answers": pred_responses,
                    "Ground Truth": ref_responses,
                },
            )

        self.experiment_status = ExperimentStatus("running")
        self.start_datetime = datetime.now()
        result = None
        try:
            if not self.tuner:
                # Select default tuner if one is not specified
                if is_ray_installed():
                    from nomadic.tuner.ray import RayTuneParamTuner

                    self.tuner = RayTuneParamTuner(
                        param_fn=default_param_function,
                        param_dict=self.param_dict,
                        search_method=self.search_method,
                        fixed_param_dict=self.fixed_param_dict,
                        current_param_dict=self.current_param_dict,
                        show_progress=True,
                    )
                else:
                    from nomadic.tuner import FlamlParamTuner

                    self.tuner = FlamlParamTuner(
                        param_fn=default_param_function,
                        param_dict=self.param_dict,
                        search_method=self.search_method,
                        fixed_param_dict=self.fixed_param_dict,
                        current_param_dict=self.current_param_dict,
                        show_progress=True,
                        num_samples=-1,
                    )

            result = self.tuner.fit()
        except Exception as e:
            is_error = True
            self.experiment_status_message = (
                f"Exception: {str(e)}\n\nStack Trace:\n{traceback.format_exc()}"
            )

        self.end_datetime = datetime.now()
        self.experiment_status = (
            ExperimentStatus("finished_success")
            if not is_error
            else ExperimentStatus("finished_error")
        )
        self.tuned_result = result or TunedResult(
            run_results=[RunResult(score=-1, params={}, metadata={})],
            best_idx=0,
        )
        return self.tuned_result

    def save_experiment(self, folder_path: Path):
        file_name = (
            f"/experiment_{self.start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(folder_path + file_name, "w") as file:
            file.write(self.model_dump_json(exclude=("model", "evaluator")))
