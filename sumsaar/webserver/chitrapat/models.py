from django.db import models

# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=500)
    content = models.TextField()
    summary = models.TextField(blank=True, null=True)
    url = models.URLField(max_length=1000, unique=True)
    source = models.CharField(max_length=100, blank=True, null=True)
    published_date = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-published_date', '-created_at']
