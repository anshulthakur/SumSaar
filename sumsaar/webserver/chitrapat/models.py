from django.db import models
from django.db.models import Q
from django.utils.safestring import mark_safe
import markdown
from django_mongodb_backend.fields import ObjectIdAutoField

# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=500)
    content = models.TextField()
    keywords = models.JSONField(default=list, blank=True)
    source_urls = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-updated_at', '-created_at']

    def get_related(self, limit=5):
        """Finds related articles based on shared keywords."""
        if not self.keywords:
            return Article.objects.none()
        
        # Construct a query to find articles sharing any of the top keywords
        q = Q()
        for keyword in self.keywords[:5]:
            q |= Q(keywords__contains=keyword)
            
        return Article.objects.filter(q).exclude(id=self.id).distinct()[:limit]

    @property
    def content_html(self):
        return mark_safe(markdown.markdown(self.content))

# --- MongoDB Staging Models ---

class RawArticle(models.Model):
    """Replaces cache.json: Stores the raw scraped feed items."""
    id = ObjectIdAutoField(primary_key=True)
    feed_id = models.IntegerField(null=True) # ID from the feed loop
    title = models.CharField(max_length=1000)
    link = models.URLField(max_length=2000)
    content = models.TextField() # Raw HTML/Text
    published_date = models.DateTimeField(null=True, blank=True)
    fetched_at = models.DateTimeField(auto_now_add=True)
    db_id = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.feed_id}: {self.title}"

class SimilarityResult(models.Model):
    """Replaces similarity_results_combined.json."""
    id = ObjectIdAutoField(primary_key=True)
    reference_id = models.IntegerField() # ID of the RawArticle
    # Stores the complex nested dict: {'LSA': [...], 'Jaccard': [...]}
    scores = models.JSONField(default=dict) 
    created_at = models.DateTimeField(auto_now_add=True)

class PipelineState(models.Model):
    """Replaces progress.json."""
    id = ObjectIdAutoField(primary_key=True)
    stage = models.CharField(max_length=50)
    last_processed_index = models.JSONField(default=list) # [i, j]
    updated_at = models.DateTimeField(auto_now=True)
    clusters = models.JSONField(default=list)
    rewritten_articles = models.JSONField(default=list)
