# Nomadic
# All rights reserved - 2024

"""
This is an example of using Nomadic to perform a hyperparameter search on an LLM pipeline.

In this example, we perform hyperparameter search over a basic RAG pipeline. This pipeline
takes in the Llama 2 academic paper, answers questions on the document, and measures a 
correctness metric. We investigate tuning the following parameters:
- Chunk size
- Top k value

We perform this hyperparameter search with either Nomadic's default ParamTuner or 
RayTuneParamTuner, which uses Ray Tune for hyperparameter optimization.
"""

import argparse
import os
from pathlib import Path
import subprocess

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    aget_responses,
)
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
    QueryResponseDataset
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader
import numpy as np
from ray import tune

from nomadic.tune.tuner import ParamTuner, RayTuneParamTuner
from nomadic.tune.tuner import RunResult

# Boilerplate code for sample run
PROJECT_DIRECTORY = f"{os.path.dirname(os.path.abspath(__file__))}"
OPENAI_ENV = os.getenv('OPENAI_API_KEY')
if not OPENAI_ENV:
    print("The OpenAI environment variable is not set. Please configure env variable `OPENAI_API_KEY`.")
    exit(1)

# Helper Functions
def execute_bash_command(command: str) -> str:
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    return result.stdout

def _build_index(chunk_size, docs):
    index_out_path = f"{PROJECT_DIRECTORY}/data/storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
    if os.listdir(index_out_path) == 0:
        # parse docs
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index
        index = VectorStoreIndex(base_nodes)
        # save index to disk
        index.storage_context.persist(index_out_path)
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=index_out_path
        )
        # load index
        index = load_index_from_storage(
            storage_context,
        )
    return index


def _get_eval_batch_runner():
    evaluator_s = SemanticSimilarityEvaluator(embed_model=OpenAIEmbedding())
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )

    return eval_batch_runner

# Objective function
def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

def main(tuning_method: str, chunk_size_hp_space, top_k_hp_space):
    # Obtain RAG inputs: docs, eval_qs, and ref_response_strs
    execute_bash_command(f'mkdir -p {PROJECT_DIRECTORY}/data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "{PROJECT_DIRECTORY}/data/llama2.pdf"')
    loader = PDFReader()
    docs0 = loader.load_data(file=Path(f"{PROJECT_DIRECTORY}/data/llama2.pdf"))
    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]

    execute_bash_command(f'wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O {PROJECT_DIRECTORY}/data/llama2_eval_qr_dataset.json')
    eval_dataset = QueryResponseDataset.from_json(
        f"{PROJECT_DIRECTORY}/data/llama2_eval_qr_dataset.json"
    )
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

    # Select tuner and configure hyperparameter search space
    fixed_param_dict = {
        "docs": docs,
        "eval_qs": eval_qs[:10],
        "ref_response_strs": ref_response_strs[:10],
    }
    if tuning_method in ['default', 'ray']:
        param_dict = {
            "chunk_size": tune.grid_search(chunk_size_hp_space),
            "top_k": tune.grid_search(top_k_hp_space)
        }
        param_tuner = RayTuneParamTuner(
            param_fn=objective_function,
            param_dict=param_dict,
            fixed_param_dict=fixed_param_dict,
            show_progress=True,
        )
    else:
        raise ValueError("tuning_method must be either 'default' or 'ray'.")
    results = param_tuner.fit()

    best_result = results.best_run_result
    best_top_k = results.best_run_result.params["top_k"]
    best_chunk_size = results.best_run_result.params["chunk_size"]

    print(f"Score: {best_result.score}")
    print(f"Top-k: {best_top_k}")
    print(f"Chunk size: {best_chunk_size}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter tuning script.")
    parser.add_argument("tuning_method", type=str, choices=["default", "ray"], help="Choose tuning method: 'default' for grid search or 'ray' for random search.")
    args = parser.parse_args()
    main(args.tuning_method, [256, 512, 1024], [1, 2, 5])
    # TODO: Finish non-Ray ParamTuner