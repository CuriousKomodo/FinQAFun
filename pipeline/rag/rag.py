import os
import json
import numpy as np
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.vector_stores.elasticsearch import ElasticsearchStore
# from llama_index.core import StorageContext
from llama_index.core import Settings
# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )

from pipeline.rag.processing import process_table_into_docs

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])


class SimpleRAG:
    def __init__(self):
        self.index = None
        self.document_id = None
#         self.reranker = FlagEmbeddingReranker(
#                         top_n=5,
#                         model="BAAI/bge-reranker-large",
# )

    def create_index(self, data_items):
        """Creates an Elastic Search Vector DB and index the data items as documents"""
        for data_item in data_items:
            documents = process_table_into_docs(data_item, True)

        vector_store = ElasticsearchStore(
            index_name="finqa",
            es_url="http://localhost:9200",
            es_password=os.getenv('ELASTIC_PASSWORD')
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=1,
            response_mode="tree_summarize",
        )

    def run_query(self, query):
        """Performs RAG query"""
        response = self.query_engine.query(
            query,
            filters={"doc_id": self.document_id}
        )  # Implement metadata hybrid search?
        print(response.response.dict())
        return response


def initialise_rag():
    simple_rag = SimpleRAG()
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))
    simple_rag.create_index(data_items[:2])
    return simple_rag


if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    simple_rag = initialise_rag()
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))
    answer = simple_rag.run_query(data_items[0]["question"])
    print(answer)