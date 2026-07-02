from django.contrib import admin
from .models import ChatSession,Messages

# Register your models here.
admin.site.register(ChatSession)
admin.site.register(Messages)