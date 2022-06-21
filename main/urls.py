from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('sentiment/', views.result, name='result'),
    path('bullies/', views.bullies, name='bullies'),
]