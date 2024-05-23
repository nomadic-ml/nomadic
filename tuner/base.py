import os
import numpy as np
import streamlit as st
import pandas as pd
import openai
import json
from collections import defaultdict
from pydantic import BaseModel, Field, ValidationError
import altair as alt

# Define RunResult and TunedResult classes
class RunResult(BaseModel):
    score: float
    params: dict
    metadata: dict = None

class TunedResult(BaseModel):
    run_results: list
    best_idx: int

class RayTuneParamTuner(BaseModel):
    param_fn: callable
    param_dict: dict
    fixed_param_dict: dict
    run_config_dict: dict = Field(default=None, description="Run config dict for Ray Tune.")

    def fit(self) -> TunedResult:
        from ray import tune as ray_tune
        from ray.train import RunConfig
        from ray.tune.result_grid import ResultGrid

        ray_param_dict = self.param_dict

        def param_fn_wrapper(ray_param_dict: dict, fixed_param_dict: dict = None) -> dict:
            fixed_param_dict = fixed_param_dict or {}
            full_param_dict = {**fixed_param_dict, **ray_param_dict}
            tuned_result = self.param_fn(full_param_dict)
            return tuned_result.dict()

        def convert_ray_tune_run_result(result_grid: ResultGrid) -> RunResult:
            try:
                run_result = RunResult(**result_grid.metrics)
            except ValidationError:
                run_result = RunResult(score=-1, params={})
            run_result.metadata = {"timestamp": result_grid.metrics.get("timestamp") if result_grid.metrics else None}
            return run_result

        run_config = RunConfig(**self.run_config_dict) if self.run_config_dict else None
        tuner = ray_tune.Tuner(
            ray_tune.with_parameters(param_fn_wrapper, fixed_param_dict=self.fixed_param_dict),
            param_space=ray_param_dict,
            run_config=run_config,
        )

        result_grids = tuner.fit()
        all_run_results = [convert_ray_tune_run_result(result_grid) for result_grid in result_grids]

        is_current_hps_in_results = False
        for run_result in all_run_results:
            if all(item in run_result.params.items() for item in self.fixed_param_dict.items()):
                is_current_hps_in_results = True
                break
        if not is_current_hps_in_results:
            all_run_results.append(
                convert_ray_tune_run_result(
                    ray_tune.Tuner(
                        ray_tune.with_parameters(param_fn_wrapper, fixed_param_dict=self.fixed_param_dict),
                        param_space={hp_name: ray_tune.grid_search([val]) for hp_name, val in self.fixed_param_dict.items()},
                        run_config=run_config,
                    ).fit()[0]
                )
            )

        sorted_run_results = sorted(all_run_results, key=lambda x: x.score, reverse=True)
        return TunedResult(run_results=sorted_run_results, best_idx=0)

# The OpenAI API key should be set in the environment before running the app
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-api-key-placeholder"

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

hyperparameters_list = ["Learning Rate", "Batch Size", "Epochs", "Top K", "Chunk Size", "Regularization Strength"]

default_search_spaces = {
    "Learning Rate": "0.001,0.01,0.1",
    "Batch Size": "16,32,64,128",
    "Epochs": "10,20,30,50",
    "Top K": "1,2,5,10",
    "Chunk Size": "256,512,1024,2048",
    "Regularization Strength": "0.01,0.1,1,10"
}

st.set_page_config(page_title="Model Optimization", page_icon=":chart_with_upwards_trend:", layout="wide")
st.sidebar.title("Model Optimization Dashboard")

model_options = ["OpenAI", "Sagemaker"]
selected_model = st.sidebar.selectbox("Select a model", model_options, index=model_options.index("OpenAI"))

# Check if the model has changed and regenerate hyperparameters if so
if 'selected_model' not in st.session_state or st.session_state['selected_model'] != selected_model:
    hyperparameters = hyperparameters_list
    st.session_state['selected_model'] = selected_model
    st.session_state['hyperparameters'] = hyperparameters
    st.session_state['hp_search_spaces'] = {hp: default_search_spaces[hp] for hp in hyperparameters}

# Initialize hp_search_spaces in session state if not already initialized
if 'hp_search_spaces' not in st.session_state:
    st.session_state['hp_search_spaces'] = {hp: default_search_spaces[hp] for hp in hyperparameters_list}

# Initialize hp_form_content in session state if not already initialized
if 'hp_form_content' not in st.session_state:
    st.session_state['hp_form_content'] = defaultdict(dict)

# Main form
with st.form(key='experiment_form'):
    # Form for defining hyperparameters
    num_hps = st.sidebar.number_input("Number of Hyperparameters to Search", min_value=1, key="num_hps")
    hp_form_content = defaultdict()

    def get_default_search_space(hp_name):
        return default_search_spaces.get(hp_name, "")

    for i in range(1, num_hps + 1):
        col1, col2 = st.columns(2)
        hp_form_content[i] = defaultdict()
        with col1:
            hp_name = st.selectbox(f"Hyperparameter {i} Name", options=default_search_spaces.keys(), index=0, key=f"name_{i}")
            hp_form_content[i][f"name"] = hp_name
        with col2:
            default_space = st.session_state['hp_search_spaces'].get(hp_name, get_default_search_space(hp_name))
            hp_form_content[i][f"search_space"] = st.text_input(f"Hyperparameter {i} Search Space", value=default_space, key=f"search_space_{i}")
            st.session_state['hp_search_spaces'][hp_name] = hp_form_content[i][f"search_space"]

    # Optional parameters
    evaluation_metric = st.selectbox("Evaluation Metric", ["Cosine similarity", "RMSE"], key="evaluation_metric_selectbox")
    evaluation_dataset_file = st.file_uploader("Evaluation dataset", type="json", key="evaluation_dataset_file")

    current_hp_values_input = st.text_area("Current Hyperparameters (Optional)", value="{'temperature': 0.5}", key="current_hp_values_input")
    user_triggered = st.text_input("User Triggered (Optional)", key="user_triggered")
    things_of_interest = st.text_area("Things of Interest (Optional)", value="{'example_key': 'example_value'}", key="things_of_interest")
    num_simulations = st.number_input("Number of Simulations (Optional)", min_value=1, value=100, key="num_simulations")
    method_of_optimization = st.text_input("Method of Optimization (Optional)", key="method_of_optimization")

    # Submit button
    submit_button = st.form_submit_button("Submit Experiment")

def convert_string_to_int_array(range_input):
    if isinstance(range_input, list):
        return range_input
    elif isinstance(range_input, str):
        range_list = range_input.replace('[', '').replace(']', '').replace(' ', '').split(',')
        return [float(val) if '.' in val else int(val) for val in range_list]
    else:
        raise ValueError("Input must be a string or a list.")

def generate_text(model, prompt, temperature):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=temperature,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        st.error(f"An error occurred while generating text with OpenAI: {str(e)}")
        return ""

class Experiment:
    def __init__(self, name, hp_space, model_type, evaluation_data, current_hp_values=None):
        self.name = name
        self.hp_space = hp_space
        self.model_type = model_type
        self.evaluation_data = evaluation_data
        self.current_hp_values = current_hp_values if current_hp_values else {}

    def run(self):
        def openai_objective_function(params_dict):
            prompt = "What is the capital of France?"
            temperature = params_dict["temperature"]
            generated_text = generate_text("gpt-3.5-turbo", prompt, temperature)
            score = len(generated_text)
            return RunResult(score=score, params=params_dict)

        def sagemaker_objective_function(params_dict):
            prompt = "What is the capital of France?"
            generated_text = generate_text("sagemaker-model", prompt, params_dict["temperature"])
            score = len(generated_text)
            return RunResult(score=score, params=params_dict)

        try:
            if self.model_type == "OpenAI":
                param_tuner = RayTuneParamTuner(
                    param_fn=openai_objective_function,
                    param_dict=self.hp_space,
                    fixed_param_dict=self.current_hp_values,
                    run_config_dict={}
                )
            elif self.model_type == "Sagemaker":
                param_tuner = RayTuneParamTuner(
                    param_fn=sagemaker_objective_function,
                    param_dict=self.hp_space,
                    fixed_param_dict=self.current_hp_values,
                    run_config_dict={}
                )
            results = param_tuner.fit()

            hyperparameters = [result.params for result in results.run_results]
            scores = [result.score for result in results.run_results]

            df = pd.DataFrame(hyperparameters)
            df['score'] = scores

            st.write("Experiment Results:")
            st.dataframe(df)

            # Save results to session state for plotting
            st.session_state['results_df'] = df

            return results
        except Exception as e:
            st.error(f"An error occurred during the experiment: {str(e)}")
            return None

# Handle form submission
if submit_button:
    st.session_state["hp_form_content"] = hp_form_content
    with st.spinner("Running experiment..."):
        hp_content = st.session_state["hp_form_content"]
        param_dict = {hp_content[i]['name']: convert_string_to_int_array(hp_content[i]['search_space']) for i in hp_content}
        
        evaluation_data = None
        if evaluation_dataset_file is not None:
            evaluation_data = json.load(evaluation_dataset_file)

        experiment = Experiment(
            name="Hyperparameter Tuning",
            hp_space=param_dict,
            model_type=selected_model,
            evaluation_data=evaluation_data,
            current_hp_values=eval(current_hp_values_input) if current_hp_values_input else None
        )
        experiment.run()

# Plot results
if 'results_df' in st.session_state:
    df = st.session_state['results_df']

    st.sidebar.title("Filter Results")
    sliders = {}
    for param in df.columns:
        if param != 'score':
            min_val = df[param].min()
            max_val = df[param].max()
            sliders[param] = st.sidebar.slider(param, float(min_val), float(max_val), (float(min_val), float(max_val)))

    filtered_df = df.copy()
    for param, slider_vals in sliders.items():
        filtered_df = filtered_df[(filtered_df[param] >= slider_vals[0]) & (filtered_df[param] <= slider_vals[1])]

    chart = alt.Chart(filtered_df).mark_circle().encode(
        x='score',
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='score:Q'
    ).repeat(
        row=[param for param in filtered_df.columns if param != 'score']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
