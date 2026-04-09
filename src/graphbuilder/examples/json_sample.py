"""
Json Sample - Migrated from sample_fromJson.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: sample_fromJson.py
New Location: src/graphbuilder/examples/json_sample.py
"""

import json

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_first_ten_items(json_data):
    return json_data[11:20]

if __name__ == "__main__":
    input_file_path = "/home/dfrobot/ljy/GraphBuilder/data/products_wiki_zh.json"

    # Path to save the new JSON file with the first 10 items
    output_file_path = '/home/dfrobot/ljy/GraphBuilder/data/sample.json'

    # Load the original JSON file
    json_data = load_json_file(input_file_path)

    # Get the first 10 items
    first_ten_items = get_first_ten_items(json_data)

    # Save the new JSON data to a file
    save_json_file(first_ten_items, output_file_path)

    print(f"10 items saved to {output_file_path}")
