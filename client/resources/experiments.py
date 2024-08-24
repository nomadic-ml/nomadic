from typing import List, Optional
from nomadic.client.resources import APIResource
from nomadic.client.resources.experiment_results import ExperimentResults
from nomadic.client.resources.models import Models
from nomadic.experiment import Experiment

from nomadic.experiment import ExperimentStatus
from nomadic.model import Model, OpenAIModel
from nomadic.result import ExperimentResult


class Experiments(APIResource):
    def load(
        self, id: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[Experiment]:
        if not (id or name):
            raise Exception(f"Both id and name cannot be null.")
        if id:
            resp_data = self._client.request("GET", f"/experiments/{id}")
            if not resp_data:
                return None
            return self._to_experiment(resp_data)
        else:
            return next(
                (experiment for experiment in self.list() if experiment.name == name),
                None,
            )

    def list(self) -> List[Experiment]:
        resp_data = self._client.request("GET", "/experiments")
        return [self._to_experiment(d) for d in resp_data]

    def list_runs(self) -> List[ExperimentResult]:
        pass

    def register(self, experiment: Experiment):
        # Upload experiment & experiment results
        model = experiment.model
        # If provided model is None, use a placeholder model to mark.
        if not model:
            model = Model(
                name="Placeholder model - No model provided",
                api_keys={},
                client_id=None,
            )
        # If a model is provided, but that model doesn't exist in the database:
        #   1. Try registering model, then registering experiment.
        #   2. If registration of model fails, use placeholder model.
        else:
            workspace_models = Models(self._client)
            try:
                model_from_db = workspace_models.load(model.client_id)
                if model_from_db is None:
                    model_registration_resp = workspace_models.register(model)
                    if model_registration_resp is None:
                        raise Exception
                    else:
                        print(
                            "Registered model for an experiment that wasn't stored in database"
                        )
            except Exception as e:
                model = Model(
                    name="Placeholder model - Provided model couldn't be registered",
                    api_keys={},
                    client_id=None,
                )

        # Step 1: Upload experiment
        upload_data = {
            "name": experiment.name,
            "config": None,
            # TODO: Address evaluation_dataset type of nomadic.Experiment.evaluation_dataset as Dict on webapp and not List[Dict]
            "evaluation_dataset": experiment.evaluation_dataset[0],
            "evaluation_metric": get_evaluation_metric(),
            # TODO: Decide how to store and deal with hyperparameter search spaces on the Workspace vs. SDK.
            "hyperparameters": [param for param in experiment.param_dict.keys()],
            "model_registration_id": experiment.model.client_id,
        }
        response = self._client.request("POST", "/experiments", json=upload_data)
        experiment.client_id = response["id"]

        # TODO: Fix circular dependency here
        # Step 2: Upload experiment results, if any
        if experiment.experiment_result:
            workspace_experiment_results = ExperimentResults(self._client)
            workspace_experiment_results.register(
                experiment.experiment_result, experiment
            )
        return response

    def delete_registration(self, experiment: Experiment):
        return self._client.request("DELETE", f"/experiments/{experiment.client_id}")

    def delete_registration_by_id(self, id: int):
        return self._client.request("DELETE", f"/experiments/{id}")

    def _to_experiment(self, resp_data: dict) -> Experiment:
        evaluator = get_evaluator(resp_data.get("evaluation_metric"))

        # TODO: Decide how to store and deal with hyperparameter search spaces on the Workspace vs. SDK.
        param_dict = {key: None for key in resp_data.get("hyperparameters", [])}

        # Get model of experiment
        workspace_models = Models(self._client)
        try:
            model = workspace_models.load(resp_data.get("model_registration_id"))
        except Exception as e:
            # The model of the experiment may have been deleted. If so,
            # handle properly by making a fake new model to display that
            # the original model was deleted.
            model = None

        return Experiment(
            name=resp_data.get("name"),
            param_dict=param_dict,
            evaluation_dataset=[resp_data.get("evaluation_dataset")],
            evaluator=evaluator,
            model_key=resp_data.get("model_registration_key"),
            model=model,
            start_datetime=resp_data.get("created_at"),
            experiment_status=ExperimentStatus.not_started,
            client_id=resp_data.get("id"),
        )


# TODO: Get actual evaluator
def get_evaluator(evaluation_metric: str):
    from llama_index.core.evaluation import SemanticSimilarityEvaluator
    from llama_index.embeddings.openai import OpenAIEmbedding

    return SemanticSimilarityEvaluator(
        embed_model=OpenAIEmbedding(), similarity_mode=evaluation_metric
    )


# TODO: Get actual evaluation metric
def get_evaluation_metric():
    return "cosine"
