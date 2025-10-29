from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_and_evaluate, name='train_and_evaluate'),
]