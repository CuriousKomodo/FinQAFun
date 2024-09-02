import os
import json
import numpy as np
from langchain_openai import OpenAIEmbeddings

from llama_index.llms.openai import OpenAI
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
# Settings.embed_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
class SimpleRAG:
    def __init__(self, llm):
        self.llm = llm
        self.index = None
#         self.reranker = FlagEmbeddingReranker(
#                         top_n=5,
#                         model="BAAI/bge-reranker-large",
# )

    def create_index(self, data_items, index_name):
        for data_item in data_items:
            documents = process_table_into_docs(data_item, True)
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

    def generate(self, query, document_id):
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            llm=sllm,
            response_mode="tree_summarize",
        )
        response = query_engine.query(
            query,
            # filter={"doc_id": document_id}
        )  # Implement metadata hybrid search?
        print(response.response.dict())
        return response

if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))
    simple_rag = SimpleRAG(llm=sllm, )
    simple_rag.create_index(data_items[:2], index_name="test")
    answer = simple_rag.generate(data_items[0]["question"], data_items[0]["id"])
    print(answer)