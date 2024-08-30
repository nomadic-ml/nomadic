import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import requests
import numpy as np

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.evaluation.eval_utils import get_responses
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    QueryResponseDataset,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from nomadic.result import RunResult

from llama_index.core.vector_stores import SimpleVectorStore


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


def run_rag_pipeline(params_dict: Dict[str, Any], evaluator: Any = None) -> RunResult:
    """
    Run the RAG pipeline with the given parameters and evaluator.

    Args:
        params_dict (Dict[str, Any]): A dictionary containing pipeline parameters.
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
    }

    required_params = ["docs", "eval_qs", "ref_response_strs"]

    # Check for required parameters
    for param in required_params:
        if param not in params_dict:
            raise ValueError(
                f"Required parameter '{param}' is missing from params_dict"
            )

    # Merge default parameters with provided parameters
    for key, value in default_params.items():
        if key not in params_dict:
            params_dict[key] = value

    # Extract parameters
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]
    overlap = params_dict["overlap"]
    similarity_threshold = params_dict["similarity_threshold"]
    max_tokens = params_dict["max_tokens"]
    temperature = params_dict["temperature"]
    model_name = params_dict["model_name"]
    embedding_model = params_dict["embedding_model"]

    # Use SemanticSimilarityEvaluator if no evaluator is provided
    if evaluator is None:
        evaluator = SemanticSimilarityEvaluator()

    # build index
    index = _build_index(chunk_size, docs, overlap, embedding_model)

    # query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        similarity_threshold=similarity_threshold,
        max_tokens=max_tokens,
        temperature=temperature,
        model_name=model_name,
    )

    # get predicted responses
    pred_response_objs = get_responses(eval_qs, query_engine, show_progress=True)

    # run evaluator
    eval_results, pred_responses = [], []
    for _, (eval_q, pred_response, ref_response) in enumerate(
        zip(eval_qs, pred_response_objs, ref_response_strs)
    ):
        eval_results.append(
            evaluator.evaluate_response(
                eval_q, response=pred_response, reference=ref_response
            )
        )
        pred_responses.append(pred_response)

    # get mean score
    mean_score = np.array([r.score for r in eval_results]).mean()

    return RunResult(
        score=mean_score,
        params=params_dict,
        metadata={"pred_responses": pred_responses},
    )
