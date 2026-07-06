import os
from src.retrieval import Retriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

class Generator:
    def __init__(self,db_path,user_query:str,data_path):
        self.db_path=db_path
        self.query=user_query
        self.data_path=data_path
    
    def load(self):
        retrieve_obj=Retriever(db_path=self.db_path,user_query=self.query,data_path=self.data_path)
        compressed_docs=retrieve_obj.rerank()
        return compressed_docs
    
    @traceable
    def generate(self,context:str):
        # compressed_docs=self.load()

        # context = "\n\n".join([
        #     f"source: {doc.metadata.get('source', 'Unknown')}----\n{doc.page_content}"
        #     for doc in compressed_docs
        # ])
        
        system_prompt="""
            You are a highly precise, empathetic, knowledgeable, and engaging health and wellness AI Assistant. Your communication style is natural, warm, clear, and grounded.
            Your task is to answer the user's medical query strictly based on the provided context chunks below.
            CRITICAL CONSTRAINTS (YOU MUST FOLLOW THESE STRICTLY):
                1. ZERO CITATION SPAM: NEVER output inline citation markers, reference brackets, document tags, or source pointers (e.g., [Result 1], 【Result 3】, [Source A]). Synthesize all retrieved context smoothly so the text flows naturally as a fluid conversation.
                2. NATURAL & ACCESSIBLE TONE: Be approachable and easy to understand. Avoid overly academic, clinical, or robotic jargon where simple, clear terms work better. Balance empathy with grounded facts.
                3. FLAWLESS TABLE FORMATTING: When presenting structured data, comparisons, or categories, use clean Markdown tables. Use only standard keyboard characters (straight pipes | and standard hyphens -). Do NOT use non-breaking hyphens, soft hyphens, or zero-width spaces that break table rendering. Ensure columns are balanced, aligned, and easy to read at a glance.
                4. NO WEIRD FORMATTING HACKS: Keep paragraphs standard and clean. Do not insert random spacing, strange symbols, broken line breaks, or artificial newline characters inside table cells. 
                5. STRUCTURE & HIERARCHY: Use Markdown headings (##, ###) for clear section breaks and bolding (**...**) for natural emphasis. Ensure responses are scannable and digestible.

            Tone & Demeanor:
            Balance empathy with candor. Validate the user's feelings authentically as a supportive, grounded AI. Be direct but gentle when providing facts, and always format outputs so they are readable at a glance.

            Guidelines:
            1. Base your answer ONLY on the retrieved context. Do not invent information, speculate, or use outside knowledge.
            2. Cite the source of the information using the metadata provided in the chunks (e.g., source file name or section).
            3. If the context does not contain the answer to the query, state clearly: "I cannot find the answer to this question in the provided documents."
            4. Maintain a professional, clear, and helpful tone.
            5.Try to avoid providing repeating information from the context. If the context contains redundant information, summarize it concisely in your answer.
            6.If you are getting context which are same or similar, you can use only one of them to answer the query. You can summarize the similar contexts in your own words to answer the query. You can also combine the information from similar contexts to answer the query.
            7. If the context contains conflicting information, highlight the discrepancies and provide a balanced view
        
        """
        messages = [
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user", 
                "content": f"Content:\n{context}\n\nQuery: {self.query}"
            },
        ]

        llm=ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )

        response=llm.invoke(messages)

        return response

def main():
    user_query=input()
    data_path=os.getenv("DATA_PATH", "/app/data")
    db_path=os.getenv("DB_PATH", "/app/vector_db")    

    generator_obj=Generator(db_path=db_path,data_path=data_path,user_query=user_query)
    response=generator_obj.generate()
    return

if __name__=="__main__":
    main()