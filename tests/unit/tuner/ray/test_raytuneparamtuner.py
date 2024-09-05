import pytest
import json
import os
import pandas as pd

from flaml import BlendSearch

from nomadic.tuner.ray import RayTuneParamTuner, ray_tune as tune
from nomadic.result import RunResult, ExperimentResult


@pytest.fixture
def ray_tune_param_tuner_instance():
    def param_fn(hp_config):
        return RunResult(
            score=ord(hp_config["param1"]) + hp_config["param2"], params=hp_config
        )

    return RayTuneParamTuner(
        param_fn=param_fn,
        param_dict={
            "param1": tune.choice(["a", "b"]),
            "param2": tune.choice([1, 3]),
        },
        fixed_param_dict={"fixed_param": 1},
        show_progress=False,
        results_filepath="test_results.json",
    )


def test_init_param_tuner(ray_tune_param_tuner_instance):
    assert ray_tune_param_tuner_instance.param_fn is not None
    for param_name, param_obj in ray_tune_param_tuner_instance.param_dict.items():
        assert (
            param_obj.domain_str
            == {
                "param1": tune.choice(["a", "b"]),
                "param2": tune.choice([1, 3]),
            }[param_name].domain_str
        )
    assert ray_tune_param_tuner_instance.fixed_param_dict == {"fixed_param": 1}


def test_generate_param_combinations(ray_tune_param_tuner_instance):
    run_results = ray_tune_param_tuner_instance.fit().run_results
    expected_combinations = [
        {"param1": "a", "param2": 1, "fixed_param": 1},
        {"param1": "a", "param2": 3, "fixed_param": 1},
        {"param1": "b", "param2": 1, "fixed_param": 1},
        {"param1": "b", "param2": 3, "fixed_param": 1},
    ]
    assert len(run_results) == 4
    for combo in expected_combinations:
        assert any(combo == result.params for result in run_results)


def test_fit_method(ray_tune_param_tuner_instance):
    tuned_result = ray_tune_param_tuner_instance.fit()
    assert isinstance(tuned_result, ExperimentResult)
    assert len(tuned_result.run_results) == 4  # 4 combinations
    assert tuned_result.best_idx == 0
    assert tuned_result.run_results[0].score == 101.0


def test_add_entries_to_results_json_file(ray_tune_param_tuner_instance):
    try:
        os.remove(ray_tune_param_tuner_instance.results_filepath)
    except OSError:
        pass
    run_result = RunResult(score=10, params={}, metadata={})
    ray_tune_param_tuner_instance.add_entries_to_results_json_file(run_result)

    with open(ray_tune_param_tuner_instance.results_filepath, "r") as file:
        data = json.load(file)
        assert len(data) == 1
        assert data[0]["score"] == 10

    os.remove(ray_tune_param_tuner_instance.results_filepath)


def test_save_results_table(ray_tune_param_tuner_instance):
    results = [{"score": 10, "params": {"param1": "a", "param2": 1}}]
    filepath = "test_results.csv"
    ray_tune_param_tuner_instance.save_results_table(results, filepath)

    df = pd.read_csv(filepath)
    assert df.shape == (1, 2)
    assert df["score"][0] == 10

    os.remove(filepath)


if __name__ == "__main__":
    pytest.main()
