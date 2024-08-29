from typing import List, Optional

from nomadic.client.resources import APIResource
from nomadic.experiment import Experiment, ExperimentStatus
from nomadic.result import ExperimentResult, RunResult


class ExperimentResults(APIResource):
    def load(
        self, experiment: Experiment, experiment_result: ExperimentResult
    ) -> Optional[ExperimentResult]:
        resp_data = self._client.request(
            "GET",
            f"/experiments/{experiment.client_id}/{experiment_result.client_id}",
        )
        if not resp_data:
            return None
        return self._to_experiment_result(resp_data[0])

    def list(self, experiment_id: int) -> List[ExperimentResult]:
        resp_data = self._client.request("GET", f"/experiment-runs/{experiment_id}")
        return [self._to_experiment_result(d) for d in resp_data]

    def register(self, experiment: Experiment, experiment_result: ExperimentResult):
        # If given experiment doesn't exist in Client, create that first
        # TODO: Fix circular dependency here
        # if not experiment.client_id:
        #     workspace_experiments = Experiments(self._client)
        #     workspace_experiments.register(experiment)
        # TODO: Determine how to enter non discrete HP search spaces in `hyperparameters`
        upload_data = {
            "overall_score": experiment_result.best_run_result.score,
            "hyperparameters": {
                hp_name: val.categories
                for hp_name, val in experiment.param_dict.items()
                if hasattr(val, "categories")
            },
            "results": {
                "best_idx": experiment_result.best_idx,
                "run_results": [
                    run_result.get_json()
                    for run_result in experiment_result.run_results
                ],
            },
            "experiment_id": experiment.client_id,
            "status": "completed",
            "name": experiment_result.name or "Unnamed Experiment Result",
        }

        # Check if experiment result is already registered:
        #   If already registered, check if an update to the registered experiment result is required:
        #       If an update is required, update experiment result.
        #       Else, NOP.
        #   If not registered, register experiment result.
        response = None
        if experiment_result.client_id:
            experiment_result_from_client = self._client.experiment_results.load(
                experiment, experiment_result
            )
            if (
                not experiment_result_from_client
                or experiment_result != experiment_result_from_client
            ):
                method = "PUT" if experiment_result_from_client else "POST"
                response = self._client.request(
                    method, f"/experiment-runs/{experiment.client_id}", json=upload_data
                )
                if not experiment_result_from_client:
                    experiment_result.client_id = response["id"]
        else:
            response = self._client.request(
                "POST",
                f"/experiment-runs/{experiment.client_id}",
                json=upload_data,
            )
            experiment_result.client_id = response["id"]

        return response

    def delete_registration(self, experiment_result: ExperimentResult):
        response = self._client.request(
            "DELETE", f"/experiment-runs/{experiment_result.client_id}"
        )
        if not response:
            return None
        experiment_result.client_id = None
        return response

    def _to_experiment_result(self, resp_data: dict) -> ExperimentResult:
        run_results = (
            [RunResult(**r) for r in resp_data.get("results").get("run_results", {})]
            if resp_data.get("results")
            else []
        )
        return ExperimentResult(
            run_results=run_results,
            best_idx=(
                resp_data.get("results").get("best_idx", 0)
                if resp_data.get("results")
                else -1
            ),
            name=resp_data.get("name"),
            client_id=resp_data.get("id"),
        )
