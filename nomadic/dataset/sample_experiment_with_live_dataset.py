from datetime import datetime
import json
import nest_asyncio
import requests
import os

from nomadic.dataset import Dataset
from nomadic.experiment import Experiment
from nomadic.model import OpenAIModel
from nomadic.tuner import tune

from dotenv import load_dotenv

from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding

nest_asyncio.apply()

load_dotenv()

# Run a generic experiment
experiment = Experiment(
    name="Sample_Nomadic_Experiment",
    model=OpenAIModel(api_keys={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}),
    evaluation_dataset=Dataset(continuous_source={"bucket_name": "testBucket", "json_file_key": "YOUR_KEY_HERE"}),
    params={"temperature", "max_tokens"},
    evaluator=SemanticSimilarityEvaluator(embed_model=OpenAIEmbedding()),
    # current_hp_values = {...}
    # fixed_param_dict = {...}
)

results = experiment.run(
    param_dict={
        "temperature": tune.choice([0.1, 0.5, 0.9]),
        "max_tokens": tune.choice([50, 100, 200]),
    },
    evaluation_dataset_cutoff_date = datetime.fromisoformat("2023-09-01T00:00:00")
)
best_result = experiment.experiment_result.run_results[0]

print(f"Best result: {best_result.params} - Score: {best_result.score}")
