from abc import abstractmethod
from typing import Any, ClassVar, Dict, Optional, Set
from pydantic import BaseModel, Field

from llama_index.core.llms import LLM, CompletionResponse
from llama_index.llms.openai import OpenAI
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
import openai

from nomadic.result import RunResult

DEFAULT_HYPERPARAMETER_SEARCH_SPACE: Dict[str, Any] = {
    "temperature": {"type": float, "values": "[0.1,0.3,0.5,0.7,0.9]"},
    "max_tokens": {"type": int, "values": "[50,100,150,200]"},
    "top_p": {"type": float, "values": "[0.1,0.3,0.5,0.7,0.9]"},
}


class Model(BaseModel):
    """Base model"""

    api_keys: Dict[str, str] = Field(..., description="API keys needed to run model.")
    llm: Optional[LLM] = Field(default=None, description="Model to run experiment")
    name: ClassVar[str] = Field(default_value="Base Model", description="Name of model")
    required_api_keys: ClassVar[Set[str]] = Field(
        default_factory=set, description="Set of expected API keys"
    )
    hyperparameters: ClassVar[Dict] = Field(
        default=DEFAULT_HYPERPARAMETER_SEARCH_SPACE,
        description="Set of hyperparameters to tune",
    )

    def model_post_init(self, ctx):
        self._set_model()

    def _set_model(self, **kwargs):
        """Set model"""
        any_missing, missing_keys = any(
            item not in self.api_keys for item in self.required_api_keys
        ), list(item not in self.api_keys for item in self.required_api_keys)
        if any_missing:
            raise NameError(
                f"The following keys are missing from the provided API keys: {missing_keys}"
            )
        # Set LLM in subclass's call

    def get_required_api_keys(self):
        return self.required_api_keys

    def get_hyperparameters(self):
        return self.hyperparameters

    @abstractmethod
    def run(self, **kwargs) -> RunResult:
        """Run model"""


class SagemakerModel(Model):
    name: ClassVar[str] = "Sagemaker"
    required_api_keys: ClassVar[Set[str]] = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "ENDPOINT_NAME",
    )

    def _set_model(
        self,
        temperature: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Set Sagemaker model"""
        super()._set_model(**kwargs)
        self.llm = SageMakerLLM(
            endpoint_name=self.api_keys["ENDPOINT_NAME"],
            aws_access_key_id=self.api_keys["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=self.api_keys["AWS_SECRET_ACCESS_KEY"],
            aws_session_token="",
            region_name=self.api_keys["AWS_DEFAULT_REGION"],  # Due to bug in LlamaIndex
            aws_region_name=self.api_keys["AWS_DEFAULT_REGION"],
            temperature=temperature,
            model_kwargs=model_kwargs,
            kwargs=kwargs,
        )

    def run(self, **kwargs) -> CompletionResponse:
        """Run Sagemaker model"""
        # Sagemaker accepts hyperparameter values within
        # the `model_kwargs` field.
        self._set_model(
            temperature=kwargs["parameters"].get("temperature", None),
            model_kwargs=kwargs["parameters"],
        )
        return self.llm.complete(
            prompt=kwargs["prompt"],
            formatted=True,
        )


DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
OPENAI_EXTRA_HYPERPARAMETER_SEARCH_SPACE: Dict[str, Any] = {
    "frequency_penalty": {"type": float, "values": "[-2.0, -1.0, 0.0, 1.0, 2.0]"},
    "presence_penalty": {"type": float, "values": "[-2.0, -1.0, 0.0, 1.0, 2.0]"},
}


class OpenAIModel(Model):
    name: ClassVar[str] = "OpenAI"
    required_api_keys: ClassVar[Set[str]] = ("OPENAI_API_KEY",)
    hyperparameters: ClassVar[Dict] = Field(
        default={
            **DEFAULT_HYPERPARAMETER_SEARCH_SPACE,
            **OPENAI_EXTRA_HYPERPARAMETER_SEARCH_SPACE,
        },
        description="Set of hyperparameters to tune",
    )
    model: Optional[str] = Field(
        default=DEFAULT_OPENAI_MODEL, description="OpenAI model to use"  # GPT-4o
    )

    def _set_model(self, **kwargs):
        """Set OpenAI model"""
        openai.api_key = self.api_keys["OPENAI_API_KEY"]
        self.llm = OpenAI(
            model=self.model,
            api_key=self.api_keys["OPENAI_API_KEY"],
        )

    def run(self, **kwargs) -> CompletionResponse:
        """Run OpenAI model"""
        if "temperature" in kwargs["parameters"]:
            self.llm = OpenAI(
                model=kwargs.get("model", DEFAULT_OPENAI_MODEL),
                api_key=self.api_keys["OPENAI_API_KEY"],
                temperature=kwargs["parameters"].get("temperature", None),
            )
        return self.llm.complete(
            prompt=kwargs["prompt"],
            **kwargs["parameters"],
        )
