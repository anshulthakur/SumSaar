from django.urls import path
from . import views

app_name = 'chitrapat'

urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<int:pk>/', views.article_detail, name='article_detail'),
    path('fetch-articles/', views.fetch_articles, name='fetch_articles'),
]