from django.shortcuts import render, get_object_or_404
from .models import Article
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .tasks import fetch_articles_task

def article_list(request):
    articles = Article.objects.all()
    return render(request, 'chitrapat/article_list.html', {'articles': articles})

def article_detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    return render(request, 'chitrapat/article_detail.html', {'article': article})

@require_POST
def fetch_articles(request):
    fetch_articles_task.delay()
    return JsonResponse({'status': 'ok', 'message': 'Article fetch initiated.'})