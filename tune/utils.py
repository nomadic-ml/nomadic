def param_fn_wrapper(
    ray_param_dict: Dict,
    fixed_param_dict: Optional[Dict] = None,
) -> Dict:
    # need a wrapper to pass in parameters to ray_tune + fixed params
    fixed_param_dict = fixed_param_dict or {}
    full_param_dict = {
        **fixed_param_dict,
        **ray_param_dict,
    }
    tuned_result = self.param_fn(full_param_dict)
    # need to convert RunResult to dict to obey
    # Ray Tune's API
    return tuned_result.model_dump()


def convert_ray_tune_run_result(result_grid: ResultGrid) -> RunResult:
    # convert dict back to RunResult (reconstruct it with metadata)
    # get the keys in RunResult, assign corresponding values in
    # result.metrics to those keys
    try:
        run_result = RunResult.model_validate(result_grid.metrics)
    except ValidationError as e:
        # Tuning function may have errored out (e.g. due to objective function erroring)
        # Handle gracefully
        run_result = RunResult(score=-1, params={})

    # add some more metadata to run_result (e.g. timestamp)
    run_result.metadata["timestamp"] = (
        result_grid.metrics["timestamp"] if result_grid.metrics else None
    )
    return run_result
