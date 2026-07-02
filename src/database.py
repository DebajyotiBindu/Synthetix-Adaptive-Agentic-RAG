import chromadb
import os 
from loader import Loader,Chunker
from embedding import Embedding

class Vectordb:
    def __init__(self,db_path,data_path):
        self.client=chromadb.PersistentClient(path=db_path)
        self.data_path=data_path
    
    def load(self):
        loader_obj=Loader(self.data_path)

        main_docs=[]
        loader_obj.load(main_docs=main_docs)

        chunk_obj=Chunker(main_docs=main_docs)
        chunks=chunk_obj.chunk()
        
        embed_obj=Embedding()
        embed_result, embedding = embed_obj.embed(chunks=chunks)
        
        return main_docs,embed_result,chunks,embedding
    
    def util(self):
        try:
            ids=[]
            docs=[]
            metadatas=[]
            
            _,embed_result,chunks,embedding=self.load()

            for i in range(len(chunks)):
                id_val=f"{i+1}"
                ids.append(id_val)

                content=chunks[i].page_content
                docs.append(content)

                metadata_val=chunks[i].metadata
                metadatas.append(metadata_val)
            
            return ids,docs,metadatas,embed_result
        
        except Exception as e:
            print(f"Error in list execution {e}")

    def save(self,db_name:str):
        try:
            collection=self.client.get_or_create_collection(
                name=db_name
            )
            print("Collection has been initialized")

            ids,docs,metadatas,embed_result=self.util()

            collection.add(
                ids=ids,
                embeddings=embed_result,
                documents=docs,
                metadatas=metadatas
            )

            print("Documents have been saved into the database")
        
        except Exception as e:
            print(f"Error in database {e}")

def main():
    data_path=r"D:\mlproject20\data"
    db_path=r"D:\mlproject20\vectordb"

    db_obj=Vectordb(db_path=db_path,data_path=data_path)
    db_name="medical_data"
    db_obj.save(db_name=db_name)

if __name__=="__main__":
    main()