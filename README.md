<p align="center">
  <img src="https://raw.githubusercontent.com/nomadic-ml/nomadic/main/assets/NomadicMLLogo.png" alt="NomadicMLLogo" width="50%">
</p>

<p align="center">
  Nomadic is an enterprise-grade toolkit by <a href="https://www.nomadicml.com/">NomadicML</a> focused on parameter search for ML teams to continuously optimize compound AI systems, from pre to post-production. Rapidly experiment and keep hyperparameters, prompts, and all aspects of your system production-ready. Teams use Nomadic to deeply understand their AI system's best levers to boost performance as it scales.
</p>

<p align="center">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/nomadic?link=https%3A%2F%2Fpypi.org%2Fproject%2Fnomadic%2F">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/nomadic?link=https%3A%2F%2Fpypi.org%2Fproject%2Fnomadic%2F">
  <img alt="Discord" src="https://img.shields.io/discord/1281121359476559996?link=https%3A%2F%2Fdiscord.gg%2FPF869aGM">
  <img alt="Static Badge" src="https://img.shields.io/badge/Pear_X-S24-green">
  <img alt="Downloads" src="https://static.pepy.tech/badge/nomadic/month">
</p>

<p align="center">
Join our <a href="https://discord.gg/mp5EJE8h">Discord</a>!
</p>

# 🗂️  Installation

You can install `nomadic` with pip (Python 3.9+ required):

```bash
pip install nomadic
```

# 📄  Documentation

Full documentation can be found here: https://docs.nomadicml.com.

Please check it out for the most up-to-date tutorials, cookbooks, SDK references, and other resources!

# Local Development

Follow the instructions below to get started on local development of the Nomadic SDK. Afterwards select the produced Python `.venv` environment in your IDE of choice.

## MacOS

```bash
make setup_dev_environment
source .venv/bin/activate
```

## Linux-Ubuntu

Coming soon!

## Build Nomadic wheel

Run:

```bash
source .venv/bin/activate # If .venv isn't already activated
make build
```

# 💻 Example Usage

## Optimizing a RAG to Boost Accuracy & Retrieval Speed by 40%

For other Quickstarts based on your application: including LLM safety, advanced RAGs, transcription/summarization (across fintech, support, healthcare),  or especially compound AI systems (multiple components > monolithic models), check out our [🍴Cookbooks](https://docs.nomadicml.com/get-started/cookbooks).

### 1. Import Nomadic Libraries and Upload OpenAI Key
``` python
import os

# Import relevant Nomadic libraries
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

# Insert your OPENAI_API_KEY below
os.environ["OPENAI_API_KEY"]= <YOUR_OPENAI_KEY>
```

### 2. Define RAG Hyperparameters for the Experiments

Say we want to explore (all of!) the following hyperparameters and search spaces to optimize a RAG performance:

| Parameter                       | Supported values                                                       | Pipeline Stage |
|---------------------------|----------------------------------------------------------------------------|----------------|
| **chunk_size**             | 128, 256, 512                                                              | Retrieval      |
| **top_k**                  | 1, 3, 5                                                                    | Retrieval      |
| **overlap**                | 50, 100, 150                                                               | Retrieval      |
| **similarity_threshold**   | 0.5, 0.7, 0.9                                                              | Retrieval      |
| **embedding_model**        | "text-embedding-ada-002", "text-embedding-curie-001"                       | Retrieval      |
| **model_name**             | "gpt-3.5-turbo", "gpt-4"                                                   | Both           |
| **temperature**            | 0.3, 0.7, 0.9                                                              | Inference      |
| **max_tokens**             | 300, 500, 700                                                              | Inference      |
| **retrieval_strategy**     | "sentence-window", "full-document"                                         | Retrieval      |
| **reranking_model**        | true, false                                                                | Inference      |
| **query_transformation**   | "rephrasing", "HyDE", "Advanced contextual refinement"                     | Retrieval      |
| **reranking_step**         | "BM25-based reranking", "dense passage retrieval (DPR)", "cross-encoder"   | Inference      |
| **reranking_model_type**   | "BM25", "DPR", "ColBERT", "cross-encoder"                                  | Retrieval      |

### Explanation of New Parameters:
- **reranking_step**: Introduces techniques for reranking the retrieved documents or chunks. This helps refine retrieval results using models such as BM25, DPR, or cross-encoders before inference.
- **reranking_model_type**: Defines the type of model used for reranking the retrieved results. Options include sparse retrieval models (BM25), dense retrieval models (DPR), and more advanced approaches like cross-encoders or ColBERT.
"sub-queries"                                    | Both           |
Then, define the search spaces for each RAG pipeline hyperparameter you want to experiment with.

```python
chunk_size = tune.choice([256, 512])
temperature = tune.choice([0.1, 0.9])
overlap = tune.choice([25])
similarity_threshold = tune.choice([50])
top_k =  tune.choice([1, 2])
max_tokens = tune.choice([100, 200])
model_name = tune.choice(["gpt-3.5-turbo", "gpt-4o"])
embedding_model = tune.choice(["text-embedding-ada-002", "text-embedding-curie-001"])
retrieval_strategy = tune.choice(["sentence-window", "auto-merging"])
```

### 3. Upload Evaluation Dataset and External Data

```python
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


```

#### 3a. Reranking Support
Nomadic supports reranking models to enhance the retrieval stage of the RAG pipeline. Reranking models, such as cross-encoders, can significantly improve the relevance of retrieved documents by scoring and reordering them based on their contextual relevance to the query. This process ensures that the most pertinent documents are provided to the language model for generating accurate and contextually appropriate responses.

To enable reranking in your experiments, specify a reranking_model in the hyperparameters and include it in the retrieval pipeline. You can experiment with different reranking models to find the one that best suits your use case. Currently supported options are: BM25, DPR, ColBERT, and Cross-encoder.

### Evaluation Metrics for Retrieval and Inference

In this demo, we use specialized evaluation metrics that work specifically well for the retrieval / inferencing stages of a RAG.
#### A. Retrieval Evaluation Metrics

- **BM25 Scoring:**
  - BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It considers term frequency (TF), inverse document frequency (IDF), document length, and other factors to compute a score. The BM25 score is used to rank the documents retrieved based on their relevance to the transformed queries. The best-retrieved documents are determined by their BM25 scores.

- **Average Retrieval Score:**
  - The average score is calculated as the mean of the BM25 scores for the best retrieval results across different queries. This score provides a measure of how well the retrieval process is performing overall.

- **Retrieval Time (in milliseconds):**
  - The total time taken to retrieve the documents is measured in milliseconds. This metric helps to evaluate the efficiency of the retrieval process, particularly in terms of speed.

#### B. Inference Evaluation Metric

- **Hallucination Score:**
  - This metric assesses the extent to which the generated response includes information not found in the context. It calculates the proportion of the predicted response tokens that match tokens found in the provided context. The score is computed as:

  `Hallucination Score = 1 - (Matching Tokens / Total Predicted Tokens)`

  - A lower hallucination score indicates that the generated response closely aligns with the provided context, while a higher score suggests the presence of hallucinated (incorrect or fabricated) information.




### 4. Run the Retrieval Experiment! 🚀
```python
# Obtain RAG inputs
docs, eval_qs, ref_response_strs = obtain_rag_inputs(pdf_url=pdf_url, eval_json=eval_json)

# Run retrieval experiment
experiment_retrieval = Experiment(
    param_fn=run_retrieval_pipeline,
    param_dict={
        "top_k": top_k,
        "model_name": model_name,
        "retrieval_strategy": retrieval_strategy,
        "embedding_model": embedding_model
    },
    fixed_param_dict={
        "docs": docs,
        "eval_qs": eval_qs[:10],
        "ref_response_strs": ref_response_strs[:10],
    },
)

# After the retrieval is done
retrieval_results = experiment_retrieval.run(param_dict={
        "top_k": top_k,
        "model_name": model_name,
        "retrieval_strategy": retrieval_strategy,
        "embedding_model": embedding_model
    })
save_run_results(retrieval_results, "run_results.json")
```

### 5. Run the Inferencing Experiment! 🚀
```python
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
        "ref_response_strs": ref_response_strs[:10],  # Make sure this matches the number of queries used in retrieval
    },
)

inference_results = experiment_inference.run(param_dict={
    "model_name": model_name,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "similarity_threshold": 0.7,
})
```

### 6. Interpret Results

Now we visualize the retrieval score (for the best run result) along with the inferencing scores for different configurations.

```python
create_retrieval_heatmap(retrieval_results)
```

![Retrieval Results](https://github.com/user-attachments/assets/91e8d760-c301-427b-975c-44520a21e22d)

Here are the results using the best-performing parameter configuration:

```python
create_inference_heatmap(inference_results)
```

![Inference Results](https://github.com/user-attachments/assets/1c9c7cc9-8b1f-4e26-8050-f7e6bd1f96ce)


# 💡 Contributing

Interested in contributing? Contributions to Nomadic as well as contributing integrations are both accepted and highly encouraged! Send questions in our [Discord]([https://discord.gg/PF869aGM).
