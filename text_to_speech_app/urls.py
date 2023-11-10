from django.urls import path
from text_to_speech_app import views

urlpatterns = [
    path('transcribe/', views.transcribe_audio, name='transcribe_audio')
]
