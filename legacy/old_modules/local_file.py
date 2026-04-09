import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
import json


def load_document_content(file_path):
    """
    Load the content of the document based on file type (PDF or JSON).
    
    Args:
        file_path (str): Path to the document.
    
    Returns:
        loader: Loader object that will be used to process the file.
    """
    if Path(file_path).suffix.lower() == '.pdf':
        print("in if: PDF")
        return PyMuPDFLoader(file_path)  # Assuming PyMuPDFLoader is defined elsewhere
    elif Path(file_path).suffix.lower() == '.json':
        print("in else: JSON")
        # For JSON, we just load it using Python's json module
        with open(file_path, 'r', encoding="utf-8") as file:
            content = json.load(file)
        return content  # Returning the loaded JSON content directly
    else:
        print("in else: Other format")
        return UnstructuredFileLoader(file_path, encoding="utf-8", mode="elements")

def process_json_to_pages(json_data, file):
    """
    Processes the JSON data to convert each item into a page representation, 
    while maintaining the structure of the data and including metadata like page numbers.
    
    Args:
        json_data (list): List of dictionaries representing the JSON data.
    
    Returns:
        list: List of pages, each with a 'page_number', 'content', and 'metadata' field.
    """
    pages = []
    page_number = 1
    page_content = ''
    metadata = {}

    # Calculate total pages (you can adjust the logic depending on the JSON structure)
    total_pages = len(json_data)
    
    for idx, item in enumerate(json_data):
        # Initialize page structure
        page_content = {"page_number": page_number, "content": {}}
        
        # Iterate through each key-value pair in the item
        for key, value in item.items():
            if isinstance(value, list):
                # If value is a list, join elements to make it a single string
                page_content["content"][key] = "\n".join(str(val) for val in value)
            else:
                # If value is not a list, store it as is
                page_content["content"][key] = value
        
        # Add metadata for the page
        page_metadata = {
            'source': f"source_{page_number}",  
            'page_number': page_number,
            'filename': file,  
            'filetype': 'json', 
            'total_pages': total_pages
        }
        
        print("JSON filename:", file)
        
        pages.append(Document(page_content = str(page_content), metadata=page_metadata))
        page_number += 1
    
    return pages

    
def get_documents_from_file_by_path(file_path, file_name):
    file_path = Path(file_path)
    if file_path.exists():
        logging.info(f'File {file_name} processing')
        # Get the file extension
        file_extension = file_path.suffix.lower()

        #try:
        # Load the content using the appropriate loader
        loader = load_document_content(file_path)
        
        if file_extension == ".pdf":
            # Process PDF using PyMuPDFLoader
            pages = loader.load()  # Assuming this has a load() method for PDFs
        elif file_extension == ".json":
            # Process JSON file and transform it into pages
            pages = process_json_to_pages(loader, file_name)
            
        else:
            # For other file types, process unstructured content
            unstructured_pages = loader.load()  # For UnstructuredFileLoader
            pages = get_pages_with_page_numbers(unstructured_pages)
        
        #except Exception as e:
        #    raise Exception(f'Error while reading the file content or metadata: {e}')
    else:
        logging.info(f'File {file_name} does not exist')
        raise Exception(f'File {file_name} does not exist')

    return file_name, pages, file_extension

def get_pages_with_page_numbers(unstructured_pages):
    pages = []
    page_number = 1
    page_content=''
    metadata = {}
    for page in unstructured_pages:
        if  'page_number' in page.metadata:
            if page.metadata['page_number']==page_number:
                page_content += page.page_content
                metadata = {'source':page.metadata['source'],'page_number':page_number, 'filename':page.metadata['filename'],
                        'filetype':page.metadata['filetype'], 'total_pages':unstructured_pages[-1].metadata['page_number']}
                
            if page.metadata['page_number']>page_number:
                page_number+=1
                if not metadata:
                    metadata = {'total_pages':unstructured_pages[-1].metadata['page_number']}
                pages.append(Document(page_content = page_content, metadata=metadata))
                page_content='' 
                
            if page == unstructured_pages[-1]:
                if not metadata:
                    metadata = {'total_pages':unstructured_pages[-1].metadata['page_number']}
                pages.append(Document(page_content = page_content, metadata=metadata))
                    
        elif page.metadata['category']=='PageBreak' and page!=unstructured_pages[0]:
            page_number+=1
            pages.append(Document(page_content = page_content, metadata=metadata))
            page_content=''
            metadata={}
        
        else:
            page_content += page.page_content
            metadata_with_custom_page_number = {'source':page.metadata['source'],
                            'page_number':1, 'filename':page.metadata['filename'],
                            'filetype':page.metadata['filetype'], 'total_pages':1}
            if page == unstructured_pages[-1]:
                    pages.append(Document(page_content = page_content, metadata=metadata_with_custom_page_number))
    return pages                