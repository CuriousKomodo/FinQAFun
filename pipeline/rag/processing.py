import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from llama_index.core.schema import TextNode

load_dotenv()

def process_table_into_docs(data_item: Dict[str, Any], with_table_parsing: bool) -> List[TextNode]:
    documents = []
    documents.extend(data_item["pre_text"])

    if with_table_parsing:
        rows = data_item["table"]
        column_names = rows[0]
        for row in rows[1:]:
            row_name = row[0]
            for idx, column in enumerate(column_names[1:]):
                info = str(f"{row_name} for {column} is {row[idx+1]}")
                documents.append(info)
    else:
        documents.append(data_item["table"])

    documents.extend(data_item["post_text"])
    documents = [TextNode(text=doc,) for doc in documents]
    return documents


if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))
    documents = process_table_into_docs(data_items[0], with_table_parsing=True)
    print(len(documents))