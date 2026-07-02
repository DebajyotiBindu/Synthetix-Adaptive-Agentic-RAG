from django.db import models

# Create your models here.
class ChatSession(models.Model):
    session_id=models.CharField(max_length=100,primary_key=True)
    title=models.CharField(max_length=200)
    created_at=models.DateTimeField(auto_now_add=True)

class Messages(models.Model):
    session=models.ForeignKey(ChatSession,on_delete=models.CASCADE,related_name="messages")
    sender=models.CharField(max_length=50)
    content=models.TextField()
    created_at=models.DateTimeField(auto_now_add=True)