from pathlib import Path
import os
import numpy as np
import streamlit as st
import pandas as pd
import openai
from collections import defaultdict
from ray import tune
from nomadic.tuner import RayTuneParamTuner

# The OpenAI API key should be set in the environment before running the app
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-api-key-placeholder"

# Initialize OpenAI client with the API key
from openai import OpenAI
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

# Form for defining hyperparameters
num_hps = st.sidebar.number_input("Number of Hyperparameters to Search", min_value=1, key="num_hps")
hp_form = st.sidebar.form("HP Space")
hp_form_content = defaultdict()

def get_default_search_space(hp_name):
    return default_search_spaces.get(hp_name, "")

with hp_form:
    for i in range(1, num_hps + 1):
        col1, col2 = st.columns(2)
        hp_form_content[i] = defaultdict()
        with col1:
            hp_name = st.selectbox(f"Hyperparameter {i} Name", options=default_search_spaces.keys(), index=0, key=f"name_{i}")
            hp_form_content[i][f"name"] = hp_name
        with col2:
            default_space = get_default_search_space(hp_name)
            hp_form_content[i][f"search_space"] = st.text_input(f"Hyperparameter {i} Search Space", value=default_space, key=f"search_space_{i}")
    col1, col2 = st.columns(2)
    with col1:
        hp_form_button = st.form_submit_button("Save Hyperparameters")
    with col2:
        hp_clear_button = st.form_submit_button("Clear Hyperparameters")
    hp_submit_button = st.form_submit_button("Submit Experiment")

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

def convert_string_to_int_array(range_input):
    if isinstance(range_input, list):
        return range_input
    elif isinstance(range_input, str):
        range_list = range_input.replace('[', '').replace(']', '').replace(' ', '').split(',')
        return [float(val) if '.' in val else int(val) for val in range_list]
    else:
        raise ValueError("Input must be a string or a list.")


class Experiment:
    def __init__(self, name, hp_space, model_type, current_hp_values=None):
        self.name = name
        self.hp_space = hp_space
        self.model_type = model_type
        self.current_hp_values = current_hp_values if current_hp_values else {}

    def run(self):
        try:
            if self.model_type == "OpenAI":
                param_tuner = RayTuneParamTuner(
                    param_fn=openai_objective_function,
                    param_dict=self.hp_space,
                    fixed_param_dict=self.current_hp_values,
                )
            elif self.model_type == "Sagemaker":
                param_tuner = RayTuneParamTuner(
                    param_fn=sagemaker_objective_function,
                    param_dict=self.hp_space,
                    fixed_param_dict=self.current_hp_values,
                )
            results = param_tuner.fit()
            return results
        except Exception as e:
            print(f"An error occurred during the experiment: {str(e)}")
            return None

if hp_form_button:
    st.session_state["hp_form_content"] = hp_form_content

# Select evaluation metric
evaluation_metric = st.sidebar.selectbox("Evaluation Metric", ["Cosine similarity", "RMSE"], key="evaluation_metric_selectbox")
st.session_state.evaluation_metric = evaluation_metric

# File uploader for evaluation dataset
evaluation_dataset_file = st.sidebar.file_uploader("Evaluation dataset", type="json", key="evaluation_dataset_file")

# Optional parameters
current_hp_values_input = st.sidebar.text_area("Current Hyperparameters (Optional)", value="{'temperature': 0.5}", key="current_hp_values_input")
user_triggered = st.sidebar.text_input("User Triggered (Optional)", key="user_triggered")
things_of_interest = st.sidebar.text_area("Things of Interest (Optional)", value="{'example_key': 'example_value'}", key="things_of_interest")
num_simulations = st.sidebar.number_input("Number of Simulations (Optional)", min_value=1, value=100, key="num_simulations")
method_of_optimization = st.sidebar.text_input("Method of Optimization (Optional)", key="method_of_optimization")

# Validation checks for optional parameters
try:
    current_hp_values = eval(current_hp_values_input) if current_hp_values_input else None
    if current_hp_values and not isinstance(current_hp_values, dict):
        raise ValueError("Current Hyperparameters must be a dictionary.")
except Exception as e:
    st.sidebar.error(f"Invalid format for Current Hyperparameters: {str(e)}")

try:
    things_of_interest = eval(things_of_interest) if things_of_interest else None
    if things_of_interest and not isinstance(things_of_interest, dict):
        raise ValueError("Things of Interest must be a dictionary.")
except Exception as e:
    st.sidebar.error(f"Invalid format for Things of Interest: {str(e)}")

