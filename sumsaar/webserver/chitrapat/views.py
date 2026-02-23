from django.shortcuts import render, get_object_or_404
from .models import Article, RawArticle, PipelineState, SimilarityResult
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

def pipeline_dashboard(request):
    state = PipelineState.objects.first()
    raw_count = RawArticle.objects.count()
    # Fetch a subset of similarity results to avoid overloading the view
    similarity_results = SimilarityResult.objects.all()[:50]
    
    context = {
        'state': state,
        'raw_count': raw_count,
        'similarity_results': similarity_results,
    }
    return render(request, 'chitrapat/pipeline_dashboard.html', context)

def raw_article_list(request):
    articles = RawArticle.objects.all().order_by('-fetched_at')
    return render(request, 'chitrapat/raw_article_list.html', {'articles': articles})

def raw_article_detail(request, feed_id):
    article = get_object_or_404(RawArticle, feed_id=feed_id)
    return render(request, 'chitrapat/raw_article_detail.html', {'article': article})