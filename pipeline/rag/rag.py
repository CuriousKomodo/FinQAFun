import os

import numpy as np
from langchain_openai import OpenAIEmbeddings, OpenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores.elasticsearch import ElasticsearchStore
# from llama_index.core import StorageContext
from llama_index.core import Settings
# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )

from pipeline.pipeline_steps.entity_extraction import Entities
from pipeline.rag.processing import process_table_into_docs

llm = OpenAI(model="gpt-4o")
sllm = llm.as_structured_llm(output_cls=Entities)

# Initialize the embeddings model
Settings.embed_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
class SimpleRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.index = None
        self.retriever = retriever
#         self.reranker = FlagEmbeddingReranker(
#                         top_n=5,
#                         model="BAAI/bge-reranker-large",
# )

    def create_index(self, document_path, index_name):
        document = SimpleDirectoryReader(document_path).load_data()
        documents = process_table_into_docs()
        # vector_store = ElasticsearchStore(
        #     index_name=index_name,
        #     es_url="http://localhost:9200",
        #     es_password=os.getenv('ELASTIC_PASSWORD')
        # )
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #
        # self.index = VectorStoreIndex.from_documents(
        #     documents,
        #     storage_context=storage_context,
        # )
        self.index = VectorStoreIndex(documents)

    def generate(self, query):
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            llm=sllm,
            response_mode="tree_summarize",  # you can also select other modes like `compact`, `refine`
        )
        response = query_engine.query(query)
        print(response.response.dict())
        return response