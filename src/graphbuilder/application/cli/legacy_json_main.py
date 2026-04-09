"""
Legacy Json Main - Migrated from main_json.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: main_json.py
New Location: src/graphbuilder/application/cli/legacy_json_main.py
"""

from graphbuilder.core.processing.processor import extract_graph_from_file_local_file, create_source_node_graph_json
from graphbuilder.core.utils.common_functions import create_graph_database_connection
import os
from graphbuilder.infrastructure.database.neo4j_client import graphDBdataAccess
import logging
import json

#To Be Modified
uri = "bolt://localhost:7687"
userName = "neo4j"
password = "neo4j"
database = "neo4j"
model = "azure_ai_gpt_4o"
allowedNodes = "controller,sensor,board,actor,module"
allowedRelationship = "兼容"

CHUNK_DIR = os.path.join(os.path.dirname(__file__), "chunks")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "data")
#try
file_name = "/home/dfrobot/ljy/GraphBuilder/data/sample.json"


graph = create_graph_database_connection(uri, userName, password, database)   

graphDb_data_Access = graphDBdataAccess(graph)

merged_file_path = os.path.join(MERGED_DIR,file_name)
logging.info(f'File path:{merged_file_path}')

success_count,failed_count = create_source_node_graph_json(graph, model, file_name)


result = extract_graph_from_file_local_file(graph, model, merged_file_path, file_name, allowedNodes, allowedRelationship)
