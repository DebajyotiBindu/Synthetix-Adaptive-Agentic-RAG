import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from langchain_core.output_parsers import PydanticOutputParser
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from tools.tool import web_tool
from src.retrieval import Retriever
from src.generator import Generator

load_dotenv()

class ResponseStructure(BaseModel):
    reason: str = Field(description="Step-by-step evaluation of the local context against the user query.")
    score: float = Field(description="A numerical float score between 0.0 and 1.0 evaluating context quality (0.0=bad, 1.0=perfect).")

def load(db_path, data_path, query):
    retrieve_obj = Retriever(db_path=db_path, user_query=query, data_path=data_path)
    compressed_docs = retrieve_obj.rerank()
    context = "\n\n".join([
        f"source: {doc.metadata.get('source', 'Unknown')}----\n{doc.page_content}"
        for doc in compressed_docs
    ])
    return context

def orchestrator(query: str):
    db_path = r"D:\mlproject20\vectordb"
    data_path = r"D:\mlproject20\data"

    context = load(db_path=db_path, data_path=data_path, query=query)

    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")        

        MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
        client = InferenceClient(model=MODEL_ID, token=hf_token)
        
        parser = PydanticOutputParser(pydantic_object=ResponseStructure)

        base_prompt = """You are an intelligent Adaptive RAG Router. Your task is to evaluate the quality of the retrieved local context against the user's query and assign a strict numerical quality score from 0.0 to 1.0.

            Evaluation Thresholds & Routing Logic:
            - SCORE > 0.7 (GOOD QUALITY): The local chunks directly, specifically, and comprehensively answer the query.
            - SCORE 0.5 - 0.7 (MID QUALITY): The local chunks are relevant but lack specific, necessary details or breadth.
            - SCORE < 0.5 (BAD QUALITY): The local chunks are irrelevant, outdated, or fail to address the query entirely.

            {format_instructions}

            User Query: {query}

            Local Context: {context}

        """

        formatted_prompt = base_prompt.format(
            format_instructions=parser.get_format_instructions(),
            query=query,
            context=context
        )

        chat_response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.01,
            max_tokens=512
        )
        
        response_message = chat_response.choices[0].message.content
        
        decision_obj: ResponseStructure = parser.parse(response_message)
        output_score = decision_obj.score

        print(f"\n[Router Analysis]: {decision_obj.reason}")
        print(f"[Router Score]: {output_score}")

        external_content = ""
        if 0.5 <= output_score < 0.7:
            print("\n[RAG Router]: Mid-quality context detected. Fetching external web snippets...")
            external_content = web_tool(query)

        if output_score < 0.5:
            print("\n[RAG Router]: Bad-quality context detected. Fetching external web snippets...")
            external_content = web_tool(query)

        generator_obj = Generator(db_path=db_path, data_path=data_path, user_query=query)

        response = None
        if output_score < 0.5:
            response = generator_obj.generate(context=external_content)
        
        elif 0.5 <= output_score < 0.7:
            temp_context = f"{context}\n\nAdditional Data:\n{external_content}"
            response = generator_obj.generate(context=temp_context)
        
        elif output_score >= 0.7:
            response = generator_obj.generate(context=context)
        
        print("\n[Final Output]:")
        print(response.content)
        return response.content
    
    except Exception as e:
        print(f"Error in orchestrator: {e}")

if __name__ == "__main__":
    print("Enter your query:")
    query = input()
    response = orchestrator(query)