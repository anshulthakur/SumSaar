from django.db import models
from django.contrib.auth.models import User
from pgvector.django import VectorField

# Create your models here.

class RawArticle(models.Model):
    """Staging Layer: Stores the raw, unprocessed feed items."""
    url = models.URLField(max_length=2000, unique=True)
    source_data = models.JSONField(default=dict) # Stores title, content, author, published_date
    fetched_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.source_data.get('title', 'No Title')

class StoryCluster(models.Model):
    STATUS_CHOICES = [
        ('developing', 'Developing'),
        ('settled', 'Settled'),
        ('archived', 'Archived'),
    ]
    title = models.CharField(max_length=500) # Auto-generated topic label
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='developing')
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class SynthesizedArticle(models.Model):
    cluster = models.ForeignKey(StoryCluster, on_delete=models.CASCADE, related_name='articles')
    headline = models.CharField(max_length=500)
    content = models.TextField()
    facts_timeline = models.JSONField(default=list) # List of atomic facts extracted so far
    sources = models.JSONField(default=list) # List of RawArticle URLs contributing to this story
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.headline

class ArticleVector(models.Model):
    """Stores embeddings for persistent Articles (SynthesizedArticle)."""
    article = models.OneToOneField(SynthesizedArticle, on_delete=models.CASCADE, related_name='vector')
    embedding = VectorField(dimensions=384)

    def __str__(self):
        return f"Vector for {self.article.title}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    followed_topics = models.JSONField(default=list)
    interest_vector = VectorField(dimensions=384, null=True, blank=True)

class UserInteraction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    article = models.ForeignKey(SynthesizedArticle, on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=50) # click, like, bookmark, time_spent
    timestamp = models.DateTimeField(auto_now_add=True)
