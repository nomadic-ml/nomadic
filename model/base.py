from abc import abstractmethod
import copy
from typing import ClassVar, Dict, Optional, Set
from pydantic import BaseModel, Field

from llama_index.core.llms import LLM, CompletionResponse
from llama_index.llms.openai import OpenAI
from llama_index.llms.sagemaker_endpoint import SageMakerLLM
import openai

from nomadic.result import RunResult


class Model(BaseModel):
    """Base model"""

    api_keys: Dict[str, str] = Field(
        ..., description="API keys needed to run model."
    )
    llm: Optional[LLM] = Field(
        default=None, description="Model to run experiment"
    )
    name: ClassVar[str] = Field(
        default_value="Base Model", description="Name of model"
    )
    expected_keys: ClassVar[Set[str]] = Field(
        default_factory=set, description="Set of expected API keys"
    )

    def model_post_init(self, ctx):
        self._set_model()

    def _set_model(self, **kwargs):
        """Set model"""
        any_missing, missing_keys = any(
            item not in self.api_keys for item in self.expected_keys
        ), list(item not in self.api_keys for item in self.expected_keys)
        if any_missing:
            raise NameError(
                f"The following keys are missing from the provided API keys: {missing_keys}"
            )
        # Set LLM in subclass's call

    def get_required_api_keys(self):
        return self.expected_keys

    @abstractmethod
    def run(self, **kwargs) -> RunResult:
        """Run model"""


class SagemakerModel(Model):
    name: ClassVar[str] = "Sagemaker"
    expected_keys: ClassVar[Set[str]] = (
        "AWS_KNOWTEX_ACCESS_KEY_ID",
        "AWS_KNOWTEX_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "ENDPOINT_NAME",
    )

    def _set_model(self, temperature: Optional[float] = None, **kwargs):
        """Set Sagemaker model"""
        super()._set_model(**kwargs)
        self.llm = SageMakerLLM(
            endpoint_name=self.api_keys["ENDPOINT_NAME"],
            aws_access_key_id=self.api_keys["AWS_KNOWTEX_ACCESS_KEY_ID"],
            aws_secret_access_key=self.api_keys[
                "AWS_KNOWTEX_SECRET_ACCESS_KEY"
            ],
            aws_session_token="",
            region_name=self.api_keys[
                "AWS_DEFAULT_REGION"
            ],  # Due to bug in LlamaIndex
            aws_region_name=self.api_keys["AWS_DEFAULT_REGION"],
            temperature=temperature,
        )

    def run(self, **kwargs) -> CompletionResponse:
        """Run Sagemaker model"""
        if "temperature" in kwargs["parameters"]:
            self.llm = self._set_model(kwargs["parameters"]["temperature"])
        model_parameters = copy.deepcopy(kwargs["parameters"]).pop(
            "temperature", None
        )
        return self.llm.complete(
            prompt=kwargs["Instruction"],
            kwargs={"parameters": model_parameters},
        )


DEFAULT_OPENAI_MODEL: str = "gpt-3.5-turbo"


class OpenAIModel(Model):
    name: ClassVar[str] = "OpenAI"
    expected_keys: ClassVar[Set[str]] = ("OPENAI_API_KEY",)

    def _set_model(self, **kwargs):
        """Set OpenAI model"""
        openai.api_key = self.api_keys["OPENAI_API_KEY"]
        self.llm = OpenAI(
            model=kwargs.get("model", DEFAULT_OPENAI_MODEL),
            api_key=self.api_keys["OPENAI_API_KEY"],
        )

    def run(self, **kwargs) -> CompletionResponse:
        """Run OpenAI model"""
        if "temperature" in kwargs["parameters"]:
            self.llm = OpenAI(
                model=kwargs.get("model", DEFAULT_OPENAI_MODEL),
                api_key=self.api_keys["OPENAI_API_KEY"],
                temperature=kwargs["temperature"],
            )
        model_parameters = copy.deepcopy(kwargs["parameters"]).pop(
            "temperature", None
        )
        return self.llm.complete(
            prompt=kwargs["Instruction"],
            kwargs={"parameters": model_parameters},
        )
