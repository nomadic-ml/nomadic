from typing import List, Optional

from nomadic.client.resources import APIResource
from nomadic.model import Model, OpenAIModel, TogetherAIModel, SagemakerModel


class Models(APIResource):
    def load(self, id: str) -> Optional[Model]:
        if id is None:
            return None
        resp_data = self._client.request("GET", f"/models/registrations/{id}")
        if not resp_data:
            return None
        return self._to_model(resp_data[0])

    def list(self) -> List[Model]:
        resp_data = self._client.request("GET", "/models/registrations/")
        return [self._to_model(d) for d in resp_data]

    def register(self, model: Model):
        upload_data = {
            "name": model.name,
            "config": model.api_keys,
            "model_key": model.key_name,
        }

        # Check if model is already registered:
        #   If already registered, check if an update to the registered model is required:
        #       If an update is required, update model.
        #       Else, NOP.
        #   If not registered, register model.
        if model.client_id:
            model_from_client = self.load(model.client_id)
            if not model_from_client or model != model_from_client:
                method = "PUT" if model_from_client else "POST"
                response = self._client.request(
                    method, f"/models/registrations", json=upload_data
                )
                if not model_from_client:
                    model.client_id = response["id"]
                return response
        else:
            response = self._client.request(
                "POST",
                "/models/registrations",
                json=upload_data,
            )
            model.client_id = response["id"]
            return response
        return None

    def delete_registration(self, model: Model):
        response = self._client.request(
            "DELETE", f"/models/registrations/{model.client_id}"
        )
        if not response:
            return None
        model.client_id = None
        return response

    def _to_model(self, resp_data: dict) -> Model:
        model_key = resp_data.get("model_key")

        if model_key == "openai":
            return OpenAIModel(
                name=resp_data.get("name"),
                api_keys=resp_data.get("config"),
                client_id=resp_data.get("id"),
            )
        elif model_key == "together.ai":
            return TogetherAIModel(
                name=resp_data.get("name"),
                api_keys=resp_data.get("config"),
                client_id=resp_data.get("id"),
            )
        elif model_key == "sagemaker":
            return SagemakerModel(
                name=resp_data.get("name"),
                api_keys=resp_data.get("config"),
                client_id=resp_data.get("id"),
            )
        else:
            raise NotImplementedError(
                "Only OpenAI, Together.AI and Sagemaker Model types are supported"
            )
