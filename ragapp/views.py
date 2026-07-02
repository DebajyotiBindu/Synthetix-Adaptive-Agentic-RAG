from django.shortcuts import render
from src.orchestrator import orchestrator
from django.http import JsonResponse
import requests
from .models import ChatSession,Messages
import uuid

# Create your views here.
def home(request):
    return render(request,'home.html')

def chat(request):
    if request.method=="POST":
        session_id=request.POST.get('session_id')
        if not session_id:
            session_id=str(uuid.uuid4)

        query=request.POST.get('query','').strip()
        if not query:
            return JsonResponse({"error":"query cannot be found"})
        

        chat_session,created_at=ChatSession.objects.get_or_create(session_id=session_id)

        Messages.objects.create(
            session=chat_session,
            sender='user',
            content=query,
        )

        response=requests.post(
            "http://127.0.0.1:8001/chat/",
            json={
                "query":query
            }
        )

        response_text=response.json()["AI_response"]

        Messages.objects.create(
            session=chat_session,
            sender='AI',
            content=response_text,
        )

        return JsonResponse(
            {
                "response":response_text,
                "session_id":session_id
            }
        )
        
    return render(request,'chat.html')