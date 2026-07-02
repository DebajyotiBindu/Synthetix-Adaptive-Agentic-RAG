from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os 

class Loader:
    def __init__(self,path):
        self.root=path
    
    def load(self,main_docs:list):
        try:
            for files in os.listdir(self.root):
                temp_docs=[]
                if files.endswith(".pdf"):
                    file_path=os.path.join(self.root,files)

                    loader=PyMuPDFLoader(
                        file_path=file_path,
                    )

                    docs=loader.load()

                    for val in docs:
                        temp_docs.append(val)
                
                main_docs.extend(temp_docs)

            print("Data loaded successfully")
            print(main_docs[0].page_content[:300])
            print(main_docs[0].metadata)
            return
        
        except Exception as e:
            print(f"Error in loading {e}")

class Chunker:
    def __init__(self,main_docs:list):
        self.data=main_docs

    def chunk(self)->List:
        try:
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n",
                            "\n",
                            " ",
                            "."],
                is_separator_regex=False
            )
            
            print("Chunking started....")
            chunked_docs=text_splitter.split_documents(self.data)
            print("Documents have been chunked")
            print(chunked_docs[0].page_content[:100])
            return chunked_docs

        except Exception as e:
            print(f"Error in chunking {e}")
        

def main():
    root_path=r"D:\mlproject20\data"
    loader_obj=Loader(root_path)

    main_docs=[]
    loader_obj.load(main_docs=main_docs)

    chunk_obj=Chunker(main_docs=main_docs)
    chunks=chunk_obj.chunk()
    return chunks

if __name__=="__main__":
    main()