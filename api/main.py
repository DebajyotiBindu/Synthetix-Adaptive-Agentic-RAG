import os
from src.orchestrator import orchestrator
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class User(BaseModel):
    query:str

@app.get('/')
def home():
    return {"message":"Welcome"}

cache={}

@app.post('/chat/')
async def chats(user:User):
    query=user.query
    response=""
    if query in cache:
        response=cache.get(query,"Error in finding")
    else:
        response=orchestrator(query=query)
        cache[query]=response

    return {
        "user":query,
        "AI_response":response
    }