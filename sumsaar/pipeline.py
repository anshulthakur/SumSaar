import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt')

# Sample news articles
articles = [
    "Government announces new economic policy amid global uncertainty.",
    "Central bank cuts interest rates to boost the economy.",
    "Tech companies unveil latest AI-driven innovations at conference.",
    "Scientists discover potential breakthrough in cancer treatment.",
    "AI research reaches new heights with deep learning advancements.",
    "Medical researchers find new approach to treating cancer."
]

### **Preprocessing**
def preprocess(text):
    return " ".join(word_tokenize(text.lower()))  # Tokenization & lowercase

articles_processed = [preprocess(article) for article in articles]

### **Compute TF-IDF Vectorization**
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(articles_processed)

### **Compute Cosine Similarity Matrix**
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Display cosine similarity scores
import pandas as pd

df = pd.DataFrame(cosine_sim_matrix, columns=[f"Article {i}" for i in range(len(articles))], 
                  index=[f"Article {i}" for i in range(len(articles))])

print("\nCosine Similarity Matrix:\n", df)

### **Word Embeddings (Word2Vec)**
tokens = [word_tokenize(article) for article in articles_processed]
word2vec_model = Word2Vec(tokens, vector_size=50, min_count=1, workers=4)
embedding_vectors = np.array([
    np.mean([word2vec_model.wv[word] for word in tokens[i] if word in word2vec_model.wv]
            or [np.zeros(50)], axis=0) for i in range(len(tokens))
])

### **Clustering (K-Means)**
num_clusters = 2  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

### **Bucket Assignment**
news_buckets = {i: [] for i in range(num_clusters)}
for i, cluster in enumerate(clusters):
    news_buckets[cluster].append(articles[i])

### **Display Results**
for bucket, news in news_buckets.items():
    print(f"\nBucket {bucket}:")
    for story in news:
        print(f"- {story}")