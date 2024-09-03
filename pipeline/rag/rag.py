import os
import json
import re
from typing import Dict

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings

from pipeline.rag.processing import process_table_into_nodes

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])


# https://docs.llamaindex.ai/en/v0.10.17/examples/vector_stores/ElasticsearchIndexDemo.html

class SimpleRAG:
    def __init__(self):
        self.index = None
        self.document_id_to_search = None

    def create_index(self, data_item: Dict):
        """Creates an Elastic Search Vector DB and index the table information of data item."""

        nodes = process_table_into_nodes(data_item, True)
        doc_name = re.sub('[^A-Za-z0-9]+', '', data_item["id"]).lower()
        vector_store = ElasticsearchStore(
            index_name=f"finqa_{doc_name}_2",
            es_url="http://localhost:9200",
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
        )

    def run_query(self, query):
        """Performs RAG query"""
        # filters = MetadataFilters(
        #     filters=[ExactMatchFilter(key="doc_id", value=self.document_id_to_search)]
        # )
        retriever = self.index.as_retriever()
        return retriever.retrieve(str_or_query_bundle=query)


if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    simple_rag = SimpleRAG()
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))
    data_item = data_items[0]

    simple_rag.create_index(data_item)
    answer = simple_rag.run_query("What is the net cash in 2008?")
    print(answer)
    # Answer returned by RAG: The percentage change in the net cash from operating activities from 2008 to 2009 was approximately 14.9%.
    # This is clearly wrong. So we should not use RAG to perform the question answering from end-to-end.