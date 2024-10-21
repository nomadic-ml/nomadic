# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io

# Third-party imports
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# llama_index imports
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.evaluation import (
    QueryResponseDataset,
    SemanticSimilarityEvaluator,
)
from llama_index.core.evaluation.eval_utils import get_responses
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response import Response
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.readers.file import PDFReader

# Local imports
from nomadic.model import OpenAIModel
from nomadic.result import RunResult

class RunResult:
    def __init__(self, response: Response, eval_result: Dict[str, Any], visualization: Optional[bytes] = None):
        self.response = response
        self.eval_result = eval_result
        self.visualization = visualization

class GraphRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.knowledge_graph = self.build_knowledge_graph()
        self.index = self.create_index()
        self.top_results = []
        self.accuracy_scores = []
        self.explanations = []

    def build_knowledge_graph(self):
        G = nx.Graph()
        for doc in self.docs:
            G.add_node(doc.doc_id, content=doc.text)
            # Add edges based on similarity or other criteria
        return G

    def create_index(self):
        return VectorStoreIndex.from_documents(self.docs)

    def global_search(self, query: str):
        # Use community detection for global search
        communities = nx.community.greedy_modularity_communities(self.knowledge_graph)
        relevant_communities = []
        for community in communities:
            if any(query.lower() in self.knowledge_graph.nodes[node]['content'].lower() for node in community):
                relevant_communities.extend(community)
        return [self.docs[int(node)] for node in relevant_communities]

    def local_search(self, query: str):
        # Use vector search for local entity-centric search
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response.source_nodes

    def apply_prompt_tuning(self, prompt: str, strategy: str):
        if strategy == "entity_extraction":
            return f"Extract key entities from the following: {prompt}"
        elif strategy == "relationship_focus":
            return f"Focus on relationships between entities in: {prompt}"
        else:
            return prompt  # Default: no tuning

    def store_results(self, top_results, accuracy_scores, explanations):
        self.top_results = top_results
        self.accuracy_scores = accuracy_scores
        self.explanations = explanations

    def get_visualization_data(self):
        return {
            "top_results": self.top_results,
            "accuracy_scores": self.accuracy_scores,
            "explanations": self.explanations
        }


def download_file(url: str, output_path: str):
    """
    Download a file from a given URL and save it to the specified path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local path where the file should be saved.

    Raises:
        requests.exceptions.RequestException: If the download fails.
    """
    response = requests.get(url, headers={"User-Agent": "Chrome"})
    response.raise_for_status()  # Raise an error on failed download
    with open(output_path, "wb") as file:
        file.write(response.content)


def _build_index(
    chunk_size: int,
    docs: List[Document],
    overlap: int = 200,
    embedding_model: str = "text-embedding-ada-002",
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """
    Build or load a VectorStoreIndex for the given documents.

    Args:
        chunk_size (int): The size of text chunks for document parsing.
        docs (List[Document]): The list of documents to index.
        overlap (int): The amount of overlap between chunks.
        embedding_model (str): The name of the embedding model to use.
        force_rebuild (bool): If True, rebuild the index even if it exists.

    Returns:
        VectorStoreIndex: The built or loaded index.
    """
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_out_path = (
        f"{CURRENT_FILE_DIR}/data/storage_{chunk_size}_{overlap}_{embedding_model}"
    )
    vector_store_path = os.path.join(index_out_path, "vector_store.json")

    Path(index_out_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(vector_store_path) or force_rebuild:
        print(f"Creating new vector store at {vector_store_path}")
        # Create a new blank vector store
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # parse docs
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index
        index = VectorStoreIndex(base_nodes, storage_context=storage_context)
        # save index to disk
        index.storage_context.persist(persist_dir=index_out_path)
    else:
        print(f"Loading existing vector store from {vector_store_path}")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_out_path)
        # load index
        index = load_index_from_storage(storage_context)

    return index


def obtain_rag_inputs(
    pdf_url: str = "https://arxiv.org/pdf/2307.09288.pdf",
    eval_dataset_url: str = "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1",
    eval_json: Optional[Dict] = None,
) -> Tuple[List[Document], List[str], List[str]]:
    """
    Obtain the necessary inputs for RAG experiments.

    This function downloads and processes the specified PDF and evaluation dataset,
    or uses a provided JSON for evaluation.

    Args:
        pdf_url (str): URL of the PDF to be used as the document source.
        eval_dataset_url (str): URL of the evaluation dataset in JSON format (used if eval_json is None).
        eval_json (Optional[Dict]): JSON containing evaluation questions and answers (if provided, eval_dataset_url is ignored).

    Returns:
        Tuple[List[Document], List[str], List[str]]: A tuple containing:
            - docs: List of Document objects representing the downloaded PDF.
            - eval_qs: List of evaluation questions.
            - ref_response_strs: List of reference responses for evaluation.
    """
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = Path(f"{CURRENT_FILE_DIR}/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and process PDF
    pdf_filename = pdf_url.split("/")[-1]
    pdf_path = data_dir / pdf_filename

    if not pdf_path.exists():
        download_file(pdf_url, pdf_path)
    loader = PDFReader()
    docs0 = loader.load_data(file=pdf_path)
    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]

    # Process evaluation data
    if eval_json is not None:
        # Use provided JSON for evaluation
        eval_qs = list(eval_json["queries"].values())
        ref_response_strs = list(eval_json["responses"].values())
    else:
        # Download and process evaluation dataset from URL
        dataset_filename = eval_dataset_url.split("/")[-1].split("?")[
            0
        ]  # Remove query parameters
        dataset_path = data_dir / dataset_filename

        if not dataset_path.exists():
            download_file(eval_dataset_url, dataset_path)
        eval_dataset = QueryResponseDataset.from_json(dataset_path)
        eval_qs = eval_dataset.questions
        ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

    return docs, eval_qs, ref_response_strs


def apply_query_transformation(
    query: str, transformation_type: str, model_name: str = "gpt-3.5-turbo"
) -> str:
    """
    Apply a specific transformation to a query.

    Args:
        query (str): The original query.
        transformation_type (str): The type of transformation to apply.
                                   Options include "rephrasing", "HyDE", "sub-queries".
        model_name (str): The model to use for HyDE query transformation (default: gpt-3.5-turbo).

    Returns:
        str: The transformed query.
    """
    if transformation_type == "rephrasing":
        transformed_query = f"Can you explain: {query} in simpler terms?"

    elif transformation_type == "HyDE":
        hyde_transform = HyDEQueryTransform(model_name=model_name)
        transformed_query = hyde_transform(query)

    elif transformation_type == "sub-queries":
        sub_queries = [
            f"What is the main concept of {query}?",
            f"What are the applications of {query}?",
        ]
        transformed_query = " AND ".join(sub_queries)

    else:
        transformed_query = query

    return transformed_query


from llama_index.core.schema import Document, QueryBundle


from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank


def rerank_documents(
    documents: List,
    query: str,
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Document]:
    # Initialize the reranker with the model and the number of documents to rerank
    reranker = SentenceTransformerRerank(model=reranking_model, top_n=len(documents))

    # Directly pass the query as a string argument, along with the documents
    reranked_documents = reranker.postprocess_nodes(documents)

    return reranked_documents


def save_run_results(results, filename="run_results.json"):
    """Save the run results to a JSON file."""
    serializable_results = {
        "run_results": [
            {"score": run.score, "params": run.params, "metadata": run.metadata}
            for run in results.run_results
        ],
        "best_idx": results.best_idx,
    }
    with open(filename, "w") as f:
        json.dump(serializable_results, f)


def load_run_results(filename="run_results.json"):
    """Load the run results from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def get_best_run_result(loaded_results):
    """Extract the best run result from the loaded results."""
    best_idx = loaded_results["best_idx"]
    return loaded_results["run_results"][best_idx]


import time  # Import the time module to measure execution time


def run_retrieval_pipeline(params_dict: Dict[str, Any]) -> RunResult:
    """
    Run the retrieval aspect of the RAG pipeline.

    Args:
        params_dict (Dict[str, Any]): A dictionary containing parameters related to the retrieval aspect.

    Returns:
        RunResult: The result of the retrieval pipeline run.
    """
    # Separate fixed params from variable params
    fixed_params = {
        "docs": params_dict.pop("docs", []),
        "eval_qs": params_dict.pop("eval_qs", []),
        "ref_response_strs": params_dict.pop("ref_response_strs", []),
    }

    # Now params_dict only contains the variable parameters

    chunk_size = params_dict.get("chunk_size", 1000)
    top_k = params_dict.get("top_k", 3)
    overlap = params_dict.get("overlap", 200)
    similarity_threshold = params_dict.get("similarity_threshold", 0.7)
    embedding_model = params_dict.get("embedding_model", "text-embedding-ada-002")
    query_transformation = params_dict.get("query_transformation", None)
    rag_mode = params_dict.get("rag_mode", "regular")

    transformed_queries = [
        apply_query_transformation(query, query_transformation)
        for query in fixed_params["eval_qs"]
    ]

    if rag_mode == "graphrag":
        index = GraphRAG(fixed_params["docs"])
    else:
        index = _build_index(chunk_size, fixed_params["docs"], overlap, embedding_model)

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # Measure retrieval time
    start_time = time.time()  # Start timing
    retrieved_docs = get_responses(
        transformed_queries, query_engine, show_progress=True
    )
    end_time = time.time()  # End timing
    retrieval_time_ms = (end_time - start_time) * 1000  # Convert time to milliseconds

    # Use BM25 for scoring
    bm25_vectorizer = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False)
    bm25_doc_vectors = bm25_vectorizer.fit_transform(
        [doc.response for doc in retrieved_docs]
    )

    bm25_query_vectors = bm25_vectorizer.transform(transformed_queries)
    bm25_idf = bm25_vectorizer.idf_

    best_retrieval_results = []
    for i, (query, doc) in enumerate(zip(transformed_queries, retrieved_docs)):
        # Calculate BM25 score
        doc_len = len(doc.response.split())
        avg_doc_len = sum(len(d.response.split()) for d in retrieved_docs) / len(
            retrieved_docs
        )
        k1 = 1.5
        b = 0.75

        bm25_scores = []
        for j in range(len(retrieved_docs)):
            score = 0
            for t, term in enumerate(bm25_vectorizer.get_feature_names_out()):
                tf = bm25_doc_vectors[j, t]
                idf = bm25_idf[t]
                score += (
                    idf
                    * tf
                    * (k1 + 1)
                    / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                )
            bm25_scores.append(score)

        best_retrieval_results.append(
            {
                "query": query,
                "best_doc": {
                    "content": doc.response,
                    "metadata": doc.metadata,
                    "score": float(bm25_scores[i]),
                },
            }
        )

    # Sort best_retrieval_results by score in descending order
    best_retrieval_results = sorted(
        best_retrieval_results, key=lambda x: x["best_doc"]["score"], reverse=True
    )

    # Calculate average score
    avg_score = sum(
        result["best_doc"]["score"] for result in best_retrieval_results
    ) / len(best_retrieval_results)

    # Create retrieval results dictionary
    retrieval_results = {
        "best_retrieval_results": best_retrieval_results,
        "transformed_queries": transformed_queries,
        "retrieval_time_ms": retrieval_time_ms,  # Include retrieval time in milliseconds
    }

    # Save retrieval results to a file
    with open("retrieval_results.json", "w") as f:
        json.dump(retrieval_results, f)

    # Generate visualization for both GraphRAG and normal RAG
    if rag_mode == "graphrag":
        index.store_results(
            [result["best_doc"]["content"] for result in best_retrieval_results],
            [result["best_doc"]["score"] for result in best_retrieval_results],
            [f"Top result for query: {result['query']}" for result in best_retrieval_results]
        )
    visualization = visualize_rag_results(
        index,
        fixed_params["eval_qs"],
        avg_score
    )

    return RunResult(
        score=avg_score,
        params=params_dict,  # This now only contains the variable parameters
        metadata={
            "retrieval_results_file": "retrieval_results.json",
            "best_retrieval_results": best_retrieval_results,
            "retrieval_time_ms": retrieval_time_ms,  # Include retrieval time in the metadata
        },
        visualization=visualization
    )


import os
import numpy as np
import json
from typing import Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def run_inference_pipeline(params_dict: Dict[str, Any]) -> RunResult:
    """
    Run the inferencing aspect of the RAG pipeline using the best retrieval results.

    Args:
        params_dict (Dict[str, Any]): A dictionary containing parameters related to the inferencing aspect.

    Returns:
        RunResult: The result of the inferencing pipeline run.
    """
    # Extract best retrieval results from fixed_param_dict
    best_retrieval_results = params_dict.get("best_retrieval_results", [])
    if not best_retrieval_results:
        raise ValueError("No best retrieval results found in the parameters.")

    # Extract model configuration parameters dynamically from params_dict
    model_name = params_dict.get("model_name", "gpt-3.5-turbo")
    temperature = params_dict.get("temperature", 0.7)
    max_tokens = params_dict.get("max_tokens", 500)
    ref_response_strs = params_dict.get("ref_response_strs", [])

    # Get the OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize the model
    model = OpenAIModel(
        model_name=model_name, api_keys={"OPENAI_API_KEY": openai_api_key}
    )

    eval_results, pred_responses = [], []
    for best_result, ref_response in zip(best_retrieval_results, ref_response_strs):
        query = best_result["query"]
        context = best_result["best_doc"]["content"]

        # Construct the prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response using the model
        response = model.run(
            prompt=prompt,
            parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        pred_response = response.text

        # Metric 1: N-gram Precision with Context Matching
        context_tokens = set(context.split())
        pred_tokens = set(pred_response.split())

        # Count how many predicted tokens are found in the context
        matching_tokens = pred_tokens.intersection(context_tokens)
        hallucination_score = (
            1 - (len(matching_tokens) / len(pred_tokens)) if pred_tokens else 1
        )

        # Metric 2: Exact Match or Substring Match Accuracy
        exact_match_score = (
            1.0
            if ref_response.strip().lower() in pred_response.strip().lower()
            else 0.0
        )

        # Aggregate Scores with custom weights
        final_score = hallucination_score * 1 + exact_match_score * 0

        # Store results in a dictionary instead of EvaluationResult
        eval_results.append({"score": final_score})
        pred_responses.append(pred_response)

    # Calculate the mean score
    mean_score = np.mean([r["score"] for r in eval_results])

    # Create dynamic hyperparameter dictionary
    hyperparameters = {
        key: value
        for key, value in params_dict.items()
        if key not in ["best_retrieval_results", "ref_response_strs"]
    }

    # Create detailed results including hyperparameters and queries
    detailed_results = []
    for i, (best_result, eval_result) in enumerate(
        zip(best_retrieval_results, eval_results)
    ):
        detailed_results.append(
            {
                "query": best_result["query"],
                "retrieval_score": best_result["best_doc"]["score"],
                "inference_score": eval_result["score"],
                "predicted_response": pred_responses[i],
                "reference_response": ref_response_strs[i],
                "hyperparameters": hyperparameters,  # Use the dynamic hyperparameters
            }
        )

    # Save inference results to a file
    inference_results = {
        "detailed_results": detailed_results,
        "mean_score": mean_score,
    }
    with open("inference_results.json", "w") as f:
        json.dump(inference_results, f)

    return RunResult(
        score=mean_score,
        params=params_dict,
        metadata={
            "inference_results_file": "inference_results.json",
            "detailed_results": detailed_results,
        },
    )


def run_rag_pipeline(param_dict: Dict[str, Any], evaluator: Any = None) -> RunResult:
    """
    Run the RAG pipeline with the given parameters and evaluator.

    Args:
        param_dict (Dict[str, Any]): A dictionary containing pipeline parameters.
        evaluator (Any, optional): The evaluator to use for assessing responses.
                                   If None, SemanticSimilarityEvaluator is used.

    Returns:
        RunResult: The result of the RAG pipeline run, including score and metadata.
    """
    # Define default values and check for required parameters
    default_params = {
        "chunk_size": 1000,
        "top_k": 3,
        "overlap": 200,
        "similarity_threshold": 0.7,
        "max_tokens": 500,
        "temperature": 0.7,
        "model_name": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-ada-002",
        "rag_mode": "regular",
        "query_mode": "global",
        "prompt_tuning_strategy": None,
    }

    required_params = ["docs", "eval_qs", "ref_response_strs"]

    # Check for required parameters
    for param in required_params:
        if param not in param_dict:
            raise ValueError(f"Required parameter '{param}' is missing from param_dict")

    # Merge default parameters with provided parameters
    for key, value in default_params.items():
        if key not in param_dict:
            param_dict[key] = value

    # Extract parameters
    chunk_size = param_dict["chunk_size"]
    docs = param_dict["docs"]
    top_k = param_dict["top_k"]
    eval_qs = param_dict["eval_qs"]
    ref_response_strs = param_dict["ref_response_strs"]
    overlap = param_dict["overlap"]
    similarity_threshold = param_dict["similarity_threshold"]
    max_tokens = param_dict["max_tokens"]
    temperature = param_dict["temperature"]
    model_name = param_dict["model_name"]
    embedding_model = param_dict["embedding_model"]
    rag_mode = param_dict["rag_mode"]
    query_mode = param_dict["query_mode"]
    prompt_tuning_strategy = param_dict["prompt_tuning_strategy"]

    # Use SemanticSimilarityEvaluator if no evaluator is provided
    if evaluator is None:
        evaluator = SemanticSimilarityEvaluator()

    # Run retrieval pipeline
    retrieval_result = run_retrieval_pipeline(param_dict)

    # Run inference pipeline
    inference_result = run_inference_pipeline({
        **param_dict,
        "best_retrieval_results": retrieval_result.metadata["best_retrieval_results"],
    })

    # Combine scores from retrieval and inference
    combined_score = (retrieval_result.score + inference_result.score) / 2

    # Combine metadata
    combined_metadata = {
        **retrieval_result.metadata,
        **inference_result.metadata,
        "retrieval_score": retrieval_result.score,
        "inference_score": inference_result.score,
    }

    return RunResult(
        score=combined_score,
        params=param_dict,
        metadata=combined_metadata,
        visualization=retrieval_result.visualization
    )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_retrieval_heatmap(experiment_result) -> None:
    """
    Create a heatmap to visualize the average retrieval scores and retrieval times for each parameter combination.

    Args:
        experiment_result (ExperimentResult): An object containing multiple run results.
    """
    # Extract data from all run results
    all_data = []
    for run_result in experiment_result.run_results:
        # Extract the parameters and metadata
        params = run_result.params
        retrieval_time_ms = run_result.metadata.get("retrieval_time_ms", 0)
        retrieval_score = run_result.score

        # Combine parameters with results
        combined_data = {
            **params,
            "retrieval_score": retrieval_score,
            "retrieval_time_ms": retrieval_time_ms,
        }
        all_data.append(combined_data)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Generate parameter combination string dynamically
    hyperparameter_cols = [
        col for col in df.columns if col not in ["retrieval_score", "retrieval_time_ms"]
    ]

    def format_param_combination(row):
        return " | ".join(f"{k}: {v}" for k, v in row[hyperparameter_cols].items())

    df["param_combination"] = df.apply(format_param_combination, axis=1)

    # Calculate average scores for each parameter combination
    avg_scores = (
        df.groupby("param_combination")
        .agg({"retrieval_score": "mean", "retrieval_time_ms": "mean"})
        .reset_index()
    )

    # Sort by retrieval score
    avg_scores = avg_scores.sort_values(by="retrieval_score", ascending=False)

    # Pivot the DataFrame for heatmap
    pivot_df = avg_scores.set_index("param_combination")

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(20, len(pivot_df) * 0.7)
    )  # Adjust figure size for readability

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Scores and Time"},
        ax=ax,
        linewidths=1.0,  # Thicker lines for clarity
        square=False,
    )
    plt.title(
        "Heatmap of Average Retrieval Scores and Times for Each Parameter Combination",
        pad=20,
        fontsize=16,
    )
    plt.ylabel("Parameter Combination", labelpad=20, fontsize=14)
    plt.xlabel("Metrics", labelpad=20, fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=11)

    # Adjust layout
    plt.tight_layout()

    plt.show()

    # Print average scores and times across all parameter sets
    print("Average Retrieval Scores and Times Across All Parameter Sets:")
    print(f"Average Retrieval Score: {pivot_df['retrieval_score'].mean():.2f}")
    print(f"Average Retrieval Time (ms): {pivot_df['retrieval_time_ms'].mean():.2f}")


def create_inference_heatmap(experiment_result):
    # Extract data from all run results
    all_data = []
    for run_result in experiment_result.run_results:
        data = run_result.metadata["detailed_results"]
        for item in data:
            item["overall_score"] = run_result.score
        all_data.extend(data)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Create param_combination string
    hyperparameter_cols = list(all_data[0]["hyperparameters"].keys())

    def format_param_combination(row):
        return " | ".join(f"{k}: {v}" for k, v in row["hyperparameters"].items())

    df["param_combination"] = df.apply(format_param_combination, axis=1)

    # Calculate average scores for each parameter combination
    avg_scores = (
        df.groupby("param_combination")
        .agg(
            {
                "retrieval_score": "mean",
                "inference_score": "mean",
                "overall_score": "mean",
            }
        )
        .reset_index()
    )

    # Sort by overall score
    avg_scores = avg_scores.sort_values(by="overall_score", ascending=False)

    # Pivot the dataframe for heatmap
    pivot_df = avg_scores.set_index("param_combination")

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(20, len(pivot_df) * 0.7)
    )  # Increase figure size for even better readability

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
        ax=ax,
        linewidths=1.0,  # Slightly thicker lines for clarity
        square=False,
    )
    plt.title(
        "Heatmap of Average Scores for Each Parameter Combination", pad=20, fontsize=16
    )
    plt.ylabel("Parameter Combination", labelpad=20, fontsize=14)
    plt.xlabel("Score Type", labelpad=20, fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=11)

    # Adjust layout
    plt.tight_layout()

    plt.show()

    # Print average scores for each metric
    print("Average Scores Across All Parameter Sets:")
    for metric in ["inference_score", "retrieval_score", "overall_score"]:
        print(f"{metric}: {pivot_df[metric].mean():.2f}")

    return pivot_df

def visualize_rag_results(rag_object, queries, retrieval_score):
    """
    Create visualizations for RAG results including retrieved results, comparison to questions, and retrieval score.

    Args:
        rag_object: The RAG object (GraphRAG or normal RAG) containing results to visualize.
        queries (List[str]): List of queries/questions asked.
        retrieval_score (float): The overall retrieval score.

    Returns:
        bytes: PNG image of the visualization.
    """
    data = rag_object.get_visualization_data()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Accuracy scores visualization
    accuracy_scores = data['accuracy_scores'][:5]  # Limit to top 5 results
    ax1.bar(range(len(accuracy_scores)), accuracy_scores)
    ax1.set_title('Accuracy Scores for Top 5 Results')
    ax1.set_xlabel('Result Index')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracy_scores):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    ax1.text(0.95, 0.95, f'Overall Retrieval Score: {retrieval_score:.2f}',
             transform=ax1.transAxes, ha='right', va='top', fontweight='bold')

    # Top results visualization
    ax2.axis('off')
    ax2.set_title('Top 5 Retrieved Results')
    text = "\n".join([f"{i+1}. {result[:100]}..." for i, result in enumerate(data['top_results'][:5])])
    ax2.text(0, 1, text, verticalalignment='top', wrap=True)

    # Comparison of results to questions
    ax3.axis('off')
    ax3.set_title('Comparison of Results to Questions')
    comparison_text = ""
    for i, (query, result) in enumerate(zip(queries[:5], data['top_results'][:5])):
        comparison_text += f"Q{i+1}: {query}\n"
        comparison_text += f"R{i+1}: {result[:100]}...\n"
        comparison_text += f"Explanation: {data['explanations'][i][:100]}...\n\n"
    ax3.text(0, 1, comparison_text, verticalalignment='top', wrap=True)

    plt.tight_layout()

    # Save the plot to a bytes object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()
