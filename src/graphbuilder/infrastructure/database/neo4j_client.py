"""
Neo4J Client - Migrated from graphbuilder.infrastructure.database.neo4j_client.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: dbAccess.py
New Location: src/graphbuilder/infrastructure/database/neo4j_client.py
"""

import logging
from langchain_community.graphs import Neo4jGraph
from graphbuilder.domain.entities.source_node import sourceNode
import json


class graphDBdataAccess:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def update_exception_db(self, file_name, exp_msg):
        try:
            job_status = "Failed"
            result = self.get_current_status_document_node(file_name)
            is_cancelled_status = result[0]['is_cancelled']
            if bool(is_cancelled_status) == True:
                job_status = 'Cancelled'
            self.graph.query("""MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg""",
                             {"fName": file_name, "status": job_status, "error_msg": exp_msg})
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in updating document node status as failed: {error_message}")
            raise Exception(error_message)

    def create_source_node(self, obj_source_node: sourceNode):
        
        try:
            job_status = "New"
            logging.info("creating source node if does not exist")
            self.graph.query("""MERGE(d:Document {fileName :$fn}) SET d.fileSize = $fs, d.fileType = $ft ,
                            d.status = $st, d.url = $url, d.awsAccessKeyId = $awsacc_key_id, 
                            d.fileSource = $f_source, d.createdAt = $c_at, d.updatedAt = $u_at, 
                            d.processingTime = $pt, d.errorMessage = $e_message, d.nodeCount= $n_count, 
                            d.relationshipCount = $r_count, d.model= $model, d.language= $language,
                            d.is_cancelled=False, d.total_chunks=0, d.processed_chunk=0""",
                             {"fn": obj_source_node.file_name, "fs": obj_source_node.file_size, "ft": obj_source_node.file_type,
                              "st": job_status, "url": obj_source_node.url, "awsacc_key_id": obj_source_node.awsAccessKeyId,
                              "f_source": obj_source_node.file_source, "c_at": obj_source_node.created_at, "u_at": obj_source_node.created_at,
                              "pt": 0, "e_message": '', "n_count": 0, "r_count": 0, "model": obj_source_node.model,
                              "language": obj_source_node.language})
        except Exception as e:
            error_message = str(e)
            logging.info(f"error_message = {error_message}")
            self.update_exception_db(obj_source_node.file_name, error_message)
            raise Exception(error_message)

    def update_source_node(self, obj_source_node: sourceNode):
        try:
            params = {}

            # Update based on the attributes in the sourceNode
            if obj_source_node.file_name:
                params['fileName'] = obj_source_node.file_name
            if obj_source_node.status:
                params['status'] = obj_source_node.status
            if obj_source_node.created_at:
                params['createdAt'] = obj_source_node.created_at
            if obj_source_node.updated_at:
                params['updatedAt'] = obj_source_node.updated_at
            if obj_source_node.processing_time:
                params['processingTime'] = round(obj_source_node.processing_time.total_seconds(), 2)
            if obj_source_node.node_count:
                params['nodeCount'] = obj_source_node.node_count
            if obj_source_node.relationship_count:
                params['relationshipCount'] = obj_source_node.relationship_count
            if obj_source_node.model:
                params['model'] = obj_source_node.model

            param = {"props": params}
            query = "MERGE(d:Document {fileName :$props.fileName}) SET d += $props"
            logging.info("Updating source node properties")
            self.graph.query(query, param)
        except Exception as e:
            error_message = str(e)
            self.update_exception_db(obj_source_node.file_name, error_message)
            raise Exception(error_message)

    def get_source_list(self):
        """
        Get all document nodes from the database, focusing on metadata.
        """
        logging.info("Getting existing documents list from graph")
        query = "MATCH(d:Document) WHERE d.fileName IS NOT NULL RETURN d ORDER BY d.updatedAt DESC"
        result = self.graph.query(query)
        list_of_json_objects = [entry['d'] for entry in result]
        return list_of_json_objects

    def get_current_status_document_node(self, file_name):
        query = """
                MATCH(d:Document {fileName : $file_name}) RETURN d.status AS Status , d.processingTime AS processingTime, 
                d.nodeCount AS nodeCount, d.model as model, d.relationshipCount as relationshipCount,
                d.total_pages AS total_pages, d.total_chunks AS total_chunks , d.fileSize as fileSize, 
                d.is_cancelled as is_cancelled, d.processed_chunk as processed_chunk, d.fileSource as fileSource
                """
        param = {"file_name" : file_name}
        return self.execute_query(query, param)

    def execute_query(self, query, param=None):
        return self.graph.query(query, param)

    def connection_check(self):
        if self.graph:
            return "Connection Successful"

    def list_unconnected_nodes(self):
        """
        List nodes that are unconnected in the graph.
        """
        query = """
                MATCH (e:!Chunk&!Document) 
                WHERE NOT exists { (e)--(:!Chunk&!Document) }
                RETURN e {.*, elementId:elementId(e), labels:labels(e)} as e
                ORDER BY e.id ASC
                LIMIT 100
                """
        query_total_nodes = """
        MATCH (e:!Chunk&!Document) 
        WHERE NOT exists { (e)--(:!Chunk&!Document) }
        RETURN count(*) as total
        """
        nodes_list = self.execute_query(query)
        total_nodes = self.execute_query(query_total_nodes)
        return nodes_list, total_nodes[0]

    def delete_unconnected_nodes(self, unconnected_entities_list):
        entities_list = list(map(str.strip, json.loads(unconnected_entities_list)))
        query = """
        MATCH (e) WHERE elementId(e) IN $elementIds
        DETACH DELETE e
        """
        param = {"elementIds": entities_list}
        return self.execute_query(query, param)

