import json
import logging
import os
import sys
from datetime import datetime
import hashlib
from dbAccess import graphDBdataAccess
from typing import List
from entities.source_node import sourceNode
from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
from llm import generate_graphDocuments
from shared.common_fn import load_embedding_model, save_graphDocuments_in_neo4j,get_chunk_and_graphDocument,delete_uploaded_local_file, create_gcs_bucket_folder_name_hashed
import shutil
from local_file import get_documents_from_file_by_path
from langchain_community.document_loaders import WebBaseLoader
import urllib.parse

logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")
class CreateChunksofDocument:
    def __init__(self, pages: list[Document], graph: Neo4jGraph):
        self.pages = pages
        self.graph = graph

    def split_file_into_chunks(self):
        """
        Split a list of documents(file pages) into chunks of fixed size.

        Args:
            pages: A list of pages to split. Each page is a list of text strings.

        Returns:
            A list of chunks each of which is a langchain Document.
        """
        logging.info("Split file into smaller chunks")
        # number_of_chunks_allowed = int(os.environ.get('NUMBER_OF_CHUNKS_ALLOWED'))
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        
        if 'page' in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                for chunk in text_splitter.split_documents([document]):
                    chunks.append(Document(page_content=chunk.page_content, metadata={'page_number':page_number}))    
        else:
            chunks = text_splitter.split_documents(self.pages)
        return chunks
    
def create_relation_between_chunks(graph, file_name, chunks: List[Document])->list:
    logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    relationships = []
    offset=0
    for i, chunk in enumerate(chunks):
        page_content_sha1 = hashlib.sha1(chunk.page_content.encode())
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1 
        if i>0:
            #offset += len(tiktoken.encoding_for_model("gpt2").encode(chunk.page_content))
            offset += len(chunks[i-1].page_content)
        if i == 0:
            firstChunk = True
        else:
            firstChunk = False  
        metadata = {"position": position,"length": len(chunk.page_content), "content_offset":offset}
        chunk_document = Document(
            page_content=chunk.page_content, metadata=metadata
        )
        
        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk_document.page_content,
            "position": position,
            "length": chunk_document.metadata["length"],
            "f_name": file_name,
            "previous_id" : previous_chunk_id,
            "content_offset" : offset
        }
        
        if 'page_number' in chunk.metadata:
            chunk_data['page_number'] = chunk.metadata['page_number']
         
        if 'start_time' in chunk.metadata and 'end_time' in chunk.metadata:
            chunk_data['start_time'] = chunk.metadata['start_time']
            chunk_data['end_time'] = chunk.metadata['end_time'] 
               
        batch_data.append(chunk_data)
        
        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk})
        
        # create relationships between chunks
        if firstChunk:
            relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
        else:
            relationships.append({
                "type": "NEXT_CHUNK",
                "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                "current_chunk_id": current_chunk_id
            })
          
    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:Chunk {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, c.content_offset=data.content_offset
        WITH data, c
        SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
            c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
            c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
        WITH data, c
        MATCH (d:Document {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
    
    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:Document {fileName: $f_name})
        MATCH (c:Chunk {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})   
    
    query_to_create_NEXT_CHUNK_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (c:Chunk {id: relationship.current_chunk_id})
        WITH c, relationship
        MATCH (pc:Chunk {id: relationship.previous_chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
    graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})   
    
    return lst_chunks_including_hash

def update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name):
    #create embedding
    isEmbedding = os.getenv('IS_EMBEDDING')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    
    embeddings, dimension = load_embedding_model(embedding_model)
    logging.info(f'embedding model:{embeddings} and dimesion:{dimension}')
    data_for_query = []
    logging.info(f"update embedding and vector index for chunks")
    if isEmbedding.upper() == "TRUE":
        for row in chunkId_chunkDoc_list:
        # for graph_document in row['graph_doc']:
            embeddings_arr = embeddings.embed_query(row['chunk_doc'].page_content)
            # logging.info(f'Embedding list {embeddings_arr}')
                                    
            data_for_query.append({
                "chunkId": row['chunk_id'],
                "embeddings": embeddings_arr
            })
            # graph.query("""MATCH (d:Document {fileName : $fileName})
            #                MERGE (c:Chunk {id:$chunkId}) SET c.embedding = $embeddings 
            #                MERGE (c)-[:PART_OF]->(d)
            #             """,
            #             {
            #                 "fileName" : file_name,
            #                 "chunkId": row['chunk_id'],
            #                 "embeddings" : embeddings_arr
            #             }
            #             )
            # logging.info('create vector index on chunk embedding')

            graph.query("""CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                            OPTIONS {indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: 'cosine'
                            }}
                        """,
                        {
                            "dimensions" : dimension
                        }
                        )
    
    query_to_create_embedding = """
        UNWIND $data AS row
        MATCH (d:Document {fileName: $fileName})
        MERGE (c:Chunk {id: row.chunkId})
        SET c.embedding = row.embeddings
        MERGE (c)-[:PART_OF]->(d)
    """       
    graph.query(query_to_create_embedding, params={"fileName":file_name, "data":data_for_query})

def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id : list):
    batch_data = []
    logging.info("Create HAS_ENTITY relationship between chunks and entities")
    chunk_node_id_set = 'id:"{}"'
    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        for node in graph_doc_chunk_id['graph_doc'].nodes:
            query_data={
                'chunk_id': graph_doc_chunk_id['chunk_id'],
                'node_type': node.type,
                'node_id': node.id
            }
            batch_data.append(query_data)
            #node_id = node.id
            #Below query is also unable to change as parametrize because we can't make parameter of Label or node type
            #https://neo4j.com/docs/cypher-manual/current/syntax/parameters/
            #graph.query('MATCH(c:Chunk {'+chunk_node_id_set.format(graph_doc_chunk_id['chunk_id'])+'}) MERGE (n:'+ node.type +'{ id: "'+node_id+'"}) MERGE (c)-[:HAS_ENTITY]->(n)')
          
    if batch_data:
        unwind_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:Chunk {id: data.chunk_id})
                    CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                    MERGE (c)-[:HAS_ENTITY]->(n)
                """
        graph.query(unwind_query, params={"batch_data": batch_data})
        
def processing_chunks(chunkId_chunkDoc_list,graph,file_name,model,allowedNodes,allowedRelationship, node_count, rel_count):
    #create vector index and update chunk node with embedding
    update_embedding_create_vector_index( graph, chunkId_chunkDoc_list, file_name)
    logging.info("Get graph document list from models")
    graph_documents =  generate_graphDocuments(model, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)
    save_graphDocuments_in_neo4j(graph, graph_documents)
    chunks_and_graphDocuments_list = get_chunk_and_graphDocument(graph_documents, chunkId_chunkDoc_list)
    merge_relationship_between_chunk_and_entites(graph, chunks_and_graphDocuments_list)
    # return graph_documents
    
    distinct_nodes = set()
    relations = []
    for graph_document in graph_documents:
        #get distinct nodes
        for node in graph_document.nodes:
            node_id = node.id
            node_type= node.type
            if (node_id, node_type) not in distinct_nodes:
                distinct_nodes.add((node_id, node_type))
    #get all relations
    for relation in graph_document.relationships:
            relations.append(relation.type)

    node_count += len(distinct_nodes)
    rel_count += len(relations)
    print(f'node count internal func:{node_count}')
    print(f'relation count internal func:{rel_count}')
    return node_count,rel_count

def processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, is_uploaded_from_local=None, merged_file_path=None, uri=None):
  """
   Extracts a Neo4jGraph from a PDF file based on the model.
   
   Args:
   	 uri: URI of the graph to extract
     db_name : db_name is database name to connect graph db
   	 userName: Username to use for graph creation ( if None will use username from config file )
   	 password: Password to use for graph creation ( if None will use password from config file )
   	 file: File object containing the PDF file to be used
   	 model: Type of model to use ('Diffbot'or'OpenAI GPT')
   
   Returns: 
   	 Json response to API with fileName, nodeCount, relationshipCount, processingTime, 
     status and model as attributes.
  """
  start_time = datetime.now()
  graphDb_data_Access = graphDBdataAccess(graph)

  result = graphDb_data_Access.get_current_status_document_node(file_name)
  
  
  logging.info("Break down file into chunks")
  
  
  bad_chars = ['"', "\n", "'"]
  for i in range(0,len(pages)):
    text = pages[i].page_content
    for j in bad_chars:
      if j == '\n':
        text = text.replace(j, ' ')
      else:
        text = text.replace(j, '')
    pages[i]=Document(page_content=str(text), metadata=pages[i].metadata)

  create_chunks_obj = CreateChunksofDocument(pages, graph)
  
  chunks = create_chunks_obj.split_file_into_chunks()
  
  chunkId_chunkDoc_list = create_relation_between_chunks(graph,file_name,chunks)
  
  if result[0]['Status'] != 'Processing':      
    obj_source_node = sourceNode()
    status = "Processing"
    obj_source_node.file_name = file_name
    obj_source_node.status = status
    obj_source_node.total_chunks = len(chunks)
    obj_source_node.total_pages = len(pages)
    obj_source_node.model = model
    logging.info(file_name)
    logging.info(obj_source_node)
    graphDb_data_Access.update_source_node(obj_source_node)
    
    logging.info('Update the status as Processing')
    update_graph_chunk_processed = int(os.environ.get('UPDATE_GRAPH_CHUNKS_PROCESSED'))
    # selected_chunks = []
    is_cancelled_status = False
    job_status = "Completed"
    node_count = 0
    rel_count = 0
    for i in range(0, len(chunkId_chunkDoc_list), update_graph_chunk_processed):
        select_chunks_upto = i+update_graph_chunk_processed
        logging.info(f'Selected Chunks upto: {select_chunks_upto}')
        if len(chunkId_chunkDoc_list) <= select_chunks_upto:
            select_chunks_upto = len(chunkId_chunkDoc_list)
        selected_chunks = chunkId_chunkDoc_list[i:select_chunks_upto]
        result = graphDb_data_Access.get_current_status_document_node(file_name)
        keys = str(result[0].keys())
        logging.info(f"RESULT KEYS: {keys}")
        is_cancelled_status = result[0]['is_cancelled']
        logging.info(f"Value of is_cancelled : {result[0]['is_cancelled']}")
        if bool(is_cancelled_status) == True:
            job_status = "Cancelled"
            logging.info('Exit from running loop of processing file')
            exit
        else:
            node_count,rel_count = processing_chunks(selected_chunks,graph,file_name,model,allowedNodes,allowedRelationship,node_count, rel_count)
            end_time = datetime.now()
            processed_time = end_time - start_time
            
            obj_source_node = sourceNode()
            obj_source_node.file_name = file_name
            obj_source_node.updated_at = end_time
            obj_source_node.processing_time = processed_time
            obj_source_node.node_count = node_count
            obj_source_node.processed_chunk = select_chunks_upto
            obj_source_node.relationship_count = rel_count
            graphDb_data_Access.update_source_node(obj_source_node)
    
    result = graphDb_data_Access.get_current_status_document_node(file_name)
    is_cancelled_status = result[0]['is_cancelled']
    if bool(is_cancelled_status) == True:
        logging.info(f'Is_cancelled True at the end extraction')
        job_status = 'Cancelled'
    logging.info(f'Job Status at the end : {job_status}')
    end_time = datetime.now()
    processed_time = end_time - start_time
    obj_source_node = sourceNode()
    obj_source_node.file_name = file_name
    obj_source_node.status = job_status
    obj_source_node.processing_time = processed_time

    graphDb_data_Access.update_source_node(obj_source_node)
    logging.info('Updated the nodeCount and relCount properties in Document node')
    logging.info(f'file:{file_name} extraction has been completed')


    # merged_file_path have value only when file uploaded from local
    
    if is_uploaded_from_local:
        gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
        if gcs_file_cache == 'True':
            folder_name = create_gcs_bucket_folder_name_hashed(uri, file_name)
            #temporary commentation
            #delete_file_from_gcs(BUCKET_UPLOAD,folder_name,file_name)
        else:
            delete_uploaded_local_file(merged_file_path, file_name)  
        
        return {
            "fileName": file_name,
            "nodeCount": node_count,
            "relationshipCount": rel_count,
            "processingTime": round(processed_time.total_seconds(),2),
            "status" : job_status,
            "model" : model,
            "success_count" : 1
        }
    else:
        logging.info('File does not process because it\'s already in Processing status')


def get_documents_from_web_page(source_url:str):
    try:
        pages = WebBaseLoader(source_url).load()
        file_name = pages[0].metadata['title']
        return file_name, pages
    except Exception as e:
        logging.error(f"Failed To Process Web URL: {source_url}")
        logging.exception(f'Exception Stack trace: {e}')
        return None, None
        
def extract_graph_from_web_page(graph, model, source_url, allowedNodes, allowedRelationship):
    file_name, pages = get_documents_from_web_page(source_url)
    if pages==None or len(pages)==0:
        logging.exception(f'Content is not available for given URL : {source_url}')
        return None

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship)

def create_source_node_graph_dfrobot_url(graph, model, source_url, source_type):
    if "dfrobot" not in source_url:
        failed_count+=1
        message = f"Not a URL for Company : {source_url}"
        raise Exception(message)
    
    success_count=0
    failed_count=0
    lst_file_name = []
    
    pages = WebBaseLoader(source_url, verify_ssl=False).load()
    
    if pages==None or len(pages)==0:
        failed_count+=1
        message = f"Unable to read data for given url : {source_url}"
        raise Exception(message)
    try:
        obj_source_node = sourceNode()
        obj_source_node.file_type = 'text'
        obj_source_node.file_source = source_type
        obj_source_node.model = model
        obj_source_node.total_pages = 1
        obj_source_node.url = urllib.parse.unquote(source_url)
        obj_source_node.created_at = datetime.now()
        obj_source_node.file_name = pages[0].metadata['title']
        obj_source_node.language = pages[0].metadata['language'] 
        obj_source_node.file_size = sys.getsizeof(pages[0].page_content)
    
    
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.create_source_node(obj_source_node)
        lst_file_name.append({'fileName':obj_source_node.file_name,'fileSize':obj_source_node.file_size,'url':obj_source_node.url,'status':'Success'})
        success_count+=1
        return lst_file_name,success_count,failed_count
    
    except Exception as e:
        logging.error(f"Error processing source node: {str(e)}. Received metadata {str(pages[0].metadata)}")
        return None, 0, 1

def create_source_node_graph_json(graph, model, source_json_path):
    """
    Processes the provided JSON file to create a source node in a graph.
    It calculates file information and adds the corresponding node to the graph.
    The JSON content is assumed to be a list of items.

    Parameters:
    - graph: The graph object where nodes will be created.
    - model: The model associated with the node.
    - source_json_path: Path to the JSON file.

    Returns:
    - result: A dictionary with file name, size, URL, and status.
    """

    result = {'fileName': '', 'fileSize': 0, 'url': '', 'status': ''}

    try:
        # Load JSON content from the file
        with open(source_json_path, 'r', encoding='utf-8') as file:
            content = json.load(file)

        logging.info(f'Creating source node for JSON file: {source_json_path}')

        # Calculate or retrieve file details
        file_name = source_json_path # Extract file name from path
        file_size = os.path.getsize(source_json_path)  # Get the file size in bytes
        url = file_name
        total_pages = len(content)  #assuming each item is a page
        failed_count = 0
        if total_pages in [None, 0]:
            failed_count += 1
            message = f"Unable to read data for given Json file : {file_name}"
            raise Exception(message)
        else:
            
            # Create and populate the source node
            obj_source_node = sourceNode()
            obj_source_node.file_name = file_name
            obj_source_node.file_type = 'json' 
            obj_source_node.file_size = file_size
            obj_source_node.file_source = 'local'
            obj_source_node.total_pages = total_pages
            obj_source_node.model = model
            obj_source_node.url = url
            obj_source_node.created_at = datetime.now()
            obj_source_node.language = "Chinese"
            # Add the node to the graph
            graphDb_data_Access = graphDBdataAccess(graph)
            graphDb_data_Access.create_source_node(obj_source_node)

    except Exception as e:
        logging.error(f"Error while processing the JSON file: {e}")
        result['status'] = 'Failed'

    return total_pages, failed_count


def extract_graph_from_file_local_file(graph, model, merged_file_path, fileName, allowedNodes, allowedRelationship):

    logging.info(f'Process file name :{fileName}')
    file_name, pages, file_extension = get_documents_from_file_by_path(merged_file_path,fileName)
    
    if pages==None or len(pages)==0:
        raise Exception(f'File content is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, True, merged_file_path)

def merge_chunks_local(file_name, total_chunks, chunk_dir, merged_dir):
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)
    logging.info(f'Merged File Path: {merged_dir}')
    merged_file_path = os.path.join(merged_dir, file_name)
    with open(merged_file_path, "wb") as write_stream:
        for i in range(1,total_chunks+1):
            chunk_file_path = os.path.join(chunk_dir, f"{file_name}_part_{i}")
            logging.info(f'Chunk File Path While Merging Parts:{chunk_file_path}')
            with open(chunk_file_path, "rb") as chunk_file:
                shutil.copyfileobj(chunk_file, write_stream)
            os.unlink(chunk_file_path)  # Delete the individual chunk file after merging
    logging.info("Chunks merged successfully and return file size")
    file_name, pages, _ = get_documents_from_file_by_path(merged_file_path,file_name)
    pdf_total_pages = pages[0].metadata['total_pages']
    file_size = os.path.getsize(merged_file_path)
    return pdf_total_pages,file_size