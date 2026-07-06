import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langsmith import traceable


class Retriever:
    def __init__(self,db_path,user_query:str,data_path):
        self.client=chromadb.PersistentClient(path=db_path)
        self.db_path=db_path
        self.query=user_query
        self.data_path=data_path
    
    def load(self):  
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True},
        )

        return embedding

    @traceable
    def search(self,db_name:str="medical_data"):
        try:
            embeddings=self.load()

            vector_store = Chroma(
                client=self.client,
                collection_name=db_name,
                embedding_function=embeddings,
            )

            #------Uncomment this for seeing the retrieved chunks yourself-----
            #results = vector_store.similarity_search_by_vector(
            #    embedding=embeddings.embed_query(self.query), 
            #    k=30
            #)

            #for i, doc in enumerate(results, 1):
            #    print(f"Document {i}:\n{doc.page_content}\n")

            return vector_store
        
        except Exception as e:
            print(f"Error in searching {e}")
    
    @traceable
    def rerank(self):
        cross_encoder=HuggingFaceCrossEncoder(
            model_name="ms-marco-MiniLM-L-6-v2"
        )

        reranker=CrossEncoderReranker(
            model=cross_encoder,
            top_n=5
        )

        vector_store=self.search()
        base_retriever=vector_store.as_retriever(
            search_kwargs={"k": 15}
        )

        compression_retriever=ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        compressed_docs = compression_retriever.invoke(self.query)
        for i, doc in enumerate(compressed_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
        
        return compressed_docs

def main():
    data_path=os.getenv("DATA_PATH", "/app/data")
    db_path=os.getenv("DB_PATH", "/app/vector_db")
    query=input()

    retrieve_obj=Retriever(db_path=db_path,user_query=query,data_path=data_path)
    compressed_docs=retrieve_obj.rerank()
    return

if __name__=="__main__":
    main()