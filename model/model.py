from abc import abstractmethod
from typing import Dict, Optional
from pydantic import BaseModel, Field

from llama_index.core.llms import LLM
from llama_index.llms.sagemaker_endpoint import SageMakerLLM


class Model(BaseModel):
    """Base model"""

    api_keys: Dict[str, str] = Field(
        ..., description="API keys needed to run model."
    )
    llm: Optional[LLM] = Field(
        default=None, description="Model to run experiment"
    )
    expected_keys = set()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = self._set_model()

    def _set_model(self, **kwargs):
        """Set model"""
        any_missing = any(
            item not in self.api_keys for item in self.expected_keys
        )
        if any_missing:
            raise NameError(
                f"The following keys are missing from the provided API keys: {any_missing}"
            )
        # Set LLM


class SagemakerModel(Model):
    EXPECTED_KEYS = (
        "AWS_KNOWTEX_ACCESS_KEY_ID",
        "AWS_KNOWTEX_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "ENDPOINT_NAME",
    )

    def _set_model():
        def _set_model(self, **kwargs):
            """Set Sagemaker model"""
            super()._set_model(self, **kwargs)
            self.llm = SageMakerLLM(
                endpoint_name=self.api_keys["ENDPOINT_NAME"],
                aws_access_key_id=self.api_keys["AWS_KNOWTEX_ACCESS_KEY_ID"],
                aws_secret_access_key=self.api_keys[
                    "AWS_KNOWTEX_SECRET_ACCESS_KEY"
                ],
                aws_region_name=self.api_keys["AWS_DEFAULT_REGION"],
            )
