import json
import logging
from pathlib import Path
from langchain_core.documents import Document


def load_json_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If it's a list of items
    if isinstance(data, list):
        documents = []
        for i, item in enumerate(data):
            # Combine main fields into a readable page_content
            page_content_parts = [
                f"Title: {item.get('title', '')}",
                f"Keywords: {item.get('keywords', '')}",
                f"Description: {item.get('description', '')}",
                f"Introduction: {flatten_text_field(item.get('introduction'))}",
                f"Function: {flatten_text_field(item.get('function'))}",
                f"Feature: {flatten_text_field(item.get('feature'))}",
                f"Specification: {flatten_text_field(item.get('specification'))}",
                f"Distribution: {flatten_text_field(item.get('distribution'))}",
                f"Related Docs: {flatten_related_docs(item.get('related_doc'))}"
            ]
            page_content = "\n\n".join(page_content_parts)

            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "product_link": item.get("product_link", ""),
                "brand": item.get("brand", ""),
                "sku": item.get("sku", ""),
                "title": item.get("title", ""),
                "index": i,
                "filetype": "json"
            }

            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents
    else:
        raise ValueError("Expected a list of items in the JSON file.")


def flatten_text_field(field):
    if isinstance(field, list):
        result = []
        for item in field:
            if isinstance(item, dict):
                result.extend([f"{k}: {v}" for k, v in item.items()])
            elif isinstance(item, str):
                result.append(item)
        return "\n".join(result)
    return str(field)


def flatten_related_docs(field):
    if isinstance(field, list):
        return "\n".join([f"{list(d.keys())[0]}: {list(d.values())[0]}" for d in field if isinstance(d, dict)])
    return str(field)


def get_documents_from_file_by_path(file_path, file_name):
    file_path = Path(file_path)
    if file_path.exists():
        logging.info(f'file {file_name} processing')
        try:
            documents = load_json_document(file_path)
        except Exception as e:
            raise Exception(f'Error while reading the JSON file: {e}')
    else:
        logging.info(f'File {file_name} does not exist')
        raise Exception(f'File {file_name} does not exist')
    
    return file_name, documents, file_path.suffix.lower()
