"""
Legacy Llm - Migrated from graphbuilder.infrastructure.services.legacy_llm.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: llm.py
New Location: src/graphbuilder/infrastructure/services/legacy_llm.py
"""

import logging
from graphbuilder.core.utils.constants import MODEL_VERSIONS
from langchain_openai import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
from typing import List
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from graphTransformer import LLMGraphTransformer


def generate_graphDocuments(model: str, graph: Neo4jGraph, chunkId_chunkDoc_list: List, allowedNodes=None, allowedRelationship=None):
    
    if  allowedNodes is None or allowedNodes=="":
        allowedNodes =[]
    else:
        allowedNodes = allowedNodes.split(',')    
    if  allowedRelationship is None or allowedRelationship=="":   
        allowedRelationship=[]
    else:
        allowedRelationship = allowedRelationship.split(',')
    
    logging.info(f"allowedNodes: {allowedNodes}, allowedRelationship: {allowedRelationship}")

    graph_documents = []

    graph_documents = get_graph_from_llm(model,chunkId_chunkDoc_list, allowedNodes, allowedRelationship) 

    logging.info(f"graph_documents = {len(graph_documents)}")
    return graph_documents


def get_llm(model_version = "azure_ai_gpt_4o"):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model_version
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    #model_name = MODEL_VERSIONS[model_version]

    if "azure" in model_version.lower():
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )
    else:
        raise ValueError(f"Model version '{model_version}' is not supported. Only Azure models are currently supported.")
    
    logging.info(f"Model created - Model Version: {model_version}")
    return llm, model_name

def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, use_function=True
):
    futures = []
    graph_document_list = []
    if not use_function:
        node_properties = False
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        use_function_call=use_function
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list
