from django.contrib import admin
from django.urls import path,include
from ragapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('chat/',views.chat,name='chat')
]