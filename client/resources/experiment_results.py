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
            f"/experiment-runs/{experiment.client_id}/{experiment_result.client_id}",
        )
        if not resp_data:
            return None
        return _to_experiment_result(resp_data)

    def list(self, experiment_result: ExperimentResult) -> List[ExperimentResult]:
        resp_data = self._client.request(
            "GET", f"/experiment-runs{experiment_result.client_id}"
        )
        return [_to_experiment_result(d) for d in resp_data]

    def register(self, experiment_result: ExperimentResult, experiment: Experiment):
        # If given experiment doesn't exist in Client, create that first
        # TODO: Fix circular dependency here
        # if not experiment.client_id:
        #     workspace_experiments = Experiments(self._client)
        #     workspace_experiments.register(experiment)
        upload_data = {
            "overall_score": experiment_result.best_run_result.score,
            "hyperparameters": experiment_result.best_run_result.params,
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
        response = self._client.request(
            "POST", f"/experiment-runs/{experiment.client_id}", json=upload_data
        )
        experiment_result.client_id = response["id"]
        return response

    def delete_registration(self, experiment_result: ExperimentResult):
        return self._client.request(
            "DELETE", f"/experiment-runs/{experiment_result.client_id}"
        )


def _to_experiment_result(resp_data: dict) -> ExperimentResult:
    run_results = [
        RunResult(**r) for r in resp_data.get("results").get("run_results", {})
    ]
    return ExperimentResult(
        run_results=run_results,
        best_idx=resp_data.get("results").get("best_idx", 0),
        name=resp_data.get("name"),
        client_id=id,
    )
