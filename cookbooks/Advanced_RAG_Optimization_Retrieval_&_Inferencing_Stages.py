import os
from nomadic.experiment import Experiment
from nomadic.model import OpenAIModel
from nomadic.tuner import tune
from nomadic.experiment.base import Experiment, retry_with_exponential_backoff
from nomadic.experiment.rag import (
    run_rag_pipeline,
    run_retrieval_pipeline,
    run_inference_pipeline,
    obtain_rag_inputs,
    save_run_results,
    load_run_results,
    get_best_run_result,
    create_inference_heatmap
)

import pandas as pd
pd.set_option('display.max_colwidth', None)
import json

from nomadic.client import NomadicClient, ClientOptions
if "NOMADIC_API_KEY" in os.environ:
    nomadic_client = NomadicClient(ClientOptions(api_key=os.environ["NOMADIC_API_KEY"], base_url="http://204.236.146.45"))

import requests; (lambda r: r.raise_for_status() if r.status_code != 200 else print("API key is valid"))(requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}))

eval_json = {
    "queries": {
        "query1": "Describe the architecture of convolutional neural networks.",
        "query2": "What are the ethical implications of AI in healthcare?",
    },
    "responses": {
        "query1": "Convolutional neural networks consist of an input layer, convolutional layers, activation functions, pooling layers, fully connected layers, and an output layer.",
        "query2": "Ethical implications include issues of privacy, autonomy, and the potential for bias, which must be carefully managed to avoid harm.",
    }
}
pdf_url = "https://www.dropbox.com/scl/fi/sbko6nyzsuw00f2nhxa38/CS229_Lecture_Notes.pdf?rlkey=pebhb2qrdh08bnyxtus8qm11v&st=yha4ikm2&dl=1"

chunk_size = tune.choice([256, 512])
temperature = tune.choice([0.1, 0.9])
overlap = tune.choice([25])
similarity_threshold = tune.choice([50])
top_k =  tune.choice([1, 2])
max_tokens = tune.choice([100, 200])
model_name = tune.choice(["gpt-3.5-turbo", "gpt-4o"])
embedding_model = tune.choice(["text-embedding-ada-002", "text-embedding-curie-001"])
retrieval_strategy = tune.choice(["sentence-window", "auto-merging"])


# Obtain RAG inputs
docs, eval_qs, ref_response_strs = obtain_rag_inputs(pdf_url=pdf_url, eval_json=eval_json)

# Run retrieval experiment
experiment_retrieval = Experiment(
    param_fn=run_retrieval_pipeline,
    params = {"top_k", "model_name", "retrieval_strategy", "embedding_model"},
    fixed_param_dict={
        "docs": docs,
        "eval_qs": eval_qs[:10],
        "ref_response_strs": ref_response_strs[:10],
    },
    enable_logging=False,
)

# After the retrieval is done
retrieval_results = experiment_retrieval.run(param_dict={
        "top_k": top_k,
        "model_name": model_name,
        "retrieval_strategy": retrieval_strategy,
        "embedding_model": embedding_model
    })
save_run_results(retrieval_results, "run_results.json")

retrieval_results.run_results

# Load the saved results and get the best run result
loaded_results = load_run_results("run_results.json")
best_run_result = get_best_run_result(loaded_results)
best_retrieval_results = best_run_result['metadata'].get("best_retrieval_results", [])

# Run inference experiment
experiment_inference = Experiment(
    param_fn=run_inference_pipeline,
    params={"temperature","model_name", "max_tokens", "reranking_model", "similarity_threshold"},
    fixed_param_dict={
        "best_retrieval_results": best_run_result['metadata'].get("best_retrieval_results", []),
        "ref_response_strs": ref_response_strs[:10],  # Make sure this matches the number of queries used in retrieval,
        "enable_logging": True
    },
    enable_logging=False,
)

inference_results = experiment_inference.run(param_dict={
      "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "similarity_threshold": 0.7,
  })
