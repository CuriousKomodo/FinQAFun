import os
import getpass
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

# define embedding function
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)

# load documents
documents = SimpleDirectoryReader("../data/files").load_data()

vector_store = ElasticsearchStore(
    index_name="cakes",
    es_url="http://localhost:9200",
    es_password=os.getenv('ELASTIC_PASSWORD')
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# Query Data

query_engine = index.as_query_engine()
response = query_engine.query("How to make a cake?")
print(response)