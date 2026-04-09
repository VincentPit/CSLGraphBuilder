from processing import extract_graph_from_file_local_file, create_source_node_graph_json
from shared.common_fn import create_graph_database_connection
import os
from dbAccess import graphDBdataAccess
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
