import os
from langchain_huggingface import HuggingFaceEmbeddings
from loader import Loader,Chunker
from typing import List

class Embedding:
    def __init__(self):
        pass 

    def embed(self,chunks:list)->List:
        try:
            print("Embedding started....")

            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={"normalize_embeddings": True},
            )
            
            text=[chunk.page_content for chunk in chunks] #embed_documents takes a list of string as argument

            result=embedding.embed_documents(text) #List of vectors
            
            print(result[0])
            print("Embedding done")

            return result,embedding 
        
        except Exception as e:
            print(f"Error in embedding {e}")

def main():
    root_path=r"D:\mlproject20\data"
    loader_obj=Loader(root_path)

    main_docs=[]
    loader_obj.load(main_docs=main_docs)

    chunk_obj=Chunker(main_docs=main_docs)
    chunks=chunk_obj.chunk()
    
    embed_obj=Embedding()
    result, embedding = embed_obj.embed(chunks=chunks)
    return result, embedding

if __name__=="__main__":
    main()