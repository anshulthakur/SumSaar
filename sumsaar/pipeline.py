import os
import json
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
use_gpu = True
try:
    import cupy as cp
    from cuml.decomposition import TruncatedSVD
except:
    from sklearn.decomposition import TruncatedSVD
    use_gpu = False

import gensim
from gensim import corpora
from gensim.models import LdaModel

from sumsaar.settings import PROJECT_DIRS
feeds_dir = PROJECT_DIRS.get('runtime')

import logging

logger = logging.getLogger(__name__)



try:
    nltk.data.find('tokenizers/punkt')
except:
    logger.warning('Download punkt')
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/stopwords')
except:
    logger.warning('Download stopwords')
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/wordnet')
except:
    logger.warning('Download wordnet')
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except:
    logger.warning('Download punkt_tab')
    nltk.download('punkt_tab')


articles = None
article_contents = None
index_to_metadata = None 

def load_article_cache():
    global articles
    global article_contents
    global index_to_metadata

    with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
        cache = json.load(fd)

    # Sort articles by ID and store (id, title, content)
    articles = sorted(cache, key=lambda x: x['id'])
    article_contents = [article['content'] for article in articles]

    # Create mapping of index to (id, title)
    index_to_metadata = {i: (article['id'], article['title'], article['content']) for i, article in enumerate(articles)}

### **Step 1: Prepare LDA Corpus**
def prepare_lda_corpus(texts):
    """Convert preprocessed texts into a dictionary and bag-of-words corpus."""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return dictionary, corpus

### **Step 2: Train LDA Model**
def train_lda_model(corpus, dictionary, num_topics=5):
    """Train an LDA model to extract topics."""
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model

### **Step 3: Compute Topic Distribution for Each Article**
def get_topic_vectors(lda_model, corpus):
    """Convert articles into topic distribution vectors."""
    topic_vectors = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_vector = np.array([prob for _, prob in topics])  # Convert topic distribution to an array
        topic_vectors.append(topic_vector)
    return np.array(topic_vectors)

### **Step 4: Compute Cosine Similarity on Topic Vectors**
def compute_lda_similarity(topic_vectors):
    """Compute cosine similarity between articles based on topic distributions."""
    return cosine_similarity(topic_vectors)

### **Preprocessing Function**
def preprocess(text):
    """Tokenize, remove stopwords, apply stemming & lemmatization"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Apply Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Apply Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return " ".join(lemmatized_tokens)  # Return processed text


### **TF-IDF Representation**
def get_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus)

### **Bag-of-Words Representation**
def get_bow_matrix(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

### **Compute Cosine Similarity**
def get_similarity(matrix):
    return cosine_similarity(matrix)

### **Jaccard Similarity**
def get_jaccard_similarity(corpus):
    num_articles = len(corpus)
    jaccard_matrix = np.zeros((num_articles, num_articles))

    for i in range(num_articles):
        set_i = set(corpus[i].split())
        for j in range(num_articles):
            set_j = set(corpus[j].split())
            jaccard_matrix[i, j] = len(set_i & set_j) / len(set_i | set_j)

    return jaccard_matrix

### **Latent Semantic Analysis (LSA)**
def get_lsa_similarity(tfidf_matrix, n_components=100):
    logger.info("Truncated SVD")
    svd = TruncatedSVD(n_components)
    logger.info("Fitting")
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    logger.info("Cosine similarity")
    return cosine_similarity(lsa_matrix)


def get_lsa_similarity_gpu(tfidf_matrix, n_components=100):
    # Convert sparse TF-IDF to CuPy array
    X = cp.array(tfidf_matrix.toarray())
    
    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(X)
    
    # Normalize rows
    norms = cp.linalg.norm(lsa_matrix, axis=1, keepdims=True)
    lsa_matrix = lsa_matrix / norms
    
    # Cosine similarity = dot product of normalized vectors
    similarity = cp.dot(lsa_matrix, lsa_matrix.T)
    return similarity.get()  # back to NumPy if needed

### **Generate Heatmap and Save Image**
def plot_heatmap(similarity_matrix, method_name):
    plt.figure(figsize=(15, 10))
    #sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", xticklabels=articles, yticklabels=articles)
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title(f"Article Similarity Heatmap ({method_name})")
    plt.savefig(f"{feeds_dir}/similarity_heatmap_{method_name}.png")
    #plt.show()

### **Find Most Similar Articles**
def find_most_similar(similarity_matrix):
    most_similar = {}
    num_articles = len(article_contents)

    for i in range(num_articles):
        similarities = similarity_matrix[i]
        most_similar_index = np.argsort(similarities)[-2]  # Exclude self (max value at index i)
        most_similar[index_to_metadata[i]] = index_to_metadata[most_similar_index]

    return most_similar

### **Save Similarity Results to CSV with Scores**
def save_similarity_to_csv(similarity_dict, similarity_matrix, method_name):
    """Save similarity results with scores to a CSV file"""
    data = []

    for i, ((id1, title1), (id2, title2)) in enumerate(similarity_dict.items()):
        # Skip entries where the two IDs are identical (self-match)
        if id1 == id2:
            continue
        similarity_score = similarity_matrix[i, np.argsort(similarity_matrix[i])[-2]]  # Extract score
        data.append({"Article ID 1": id1, "Title 1": title1, "Similarity Score": round(similarity_score, 4), "Article ID 2": id2, "Title 2": title2})

    df = pd.DataFrame(data)
    
    filename = os.path.join(feeds_dir, f"similarity_results_{method_name}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"✅ Saved: {filename}")

### **Categorize Similarity Scores Based on Thresholds**
def categorize_similarity(similarity_matrix, strong_threshold, medium_threshold):
    """Categorizes similar articles into strong/medium based on thresholds using global mappings"""
    categorized_data = {}

    num_articles = len(article_contents)

    for i in range(num_articles):
        id1, title1, content1 = index_to_metadata[i]

        # Initialize structure for each article
        categorized_data[id1] = {
            "id": id1,
            "title": title1,
            "scores": {"strong": [], "medium": []}
        }

        for j in range(num_articles):
            if i == j:  # Skip self-match
                continue

            id2, title2, content2 = index_to_metadata[j]
            similarity_score = round(similarity_matrix[i, j], 4)

            # Assign similarity scores into categories
            if similarity_score >= strong_threshold:
                categorized_data[id1]["scores"]["strong"].append({"id": id2, "title": title2, "similarity": similarity_score})
            elif similarity_score >= medium_threshold:
                categorized_data[id1]["scores"]["medium"].append({"id": id2, "title": title2, "similarity": similarity_score})

    return categorized_data


### **Save Final JSON File**
def save_combined_json(categorized_tfidf, categorized_bow, categorized_jaccard, categorized_lsa, categorized_lda):
    combined_data = {}

    # Merge all similarity methods into one structure
    for article_id in categorized_tfidf:
        combined_data[article_id] = {
            "id": article_id,
            "title": categorized_tfidf[article_id]["title"],
            "scores": {
                "cosine": categorized_tfidf[article_id]["scores"],
                "bag-of-words": categorized_bow[article_id]["scores"],
                "Jaccard": categorized_jaccard[article_id]["scores"],
                "LSA": categorized_lsa[article_id]["scores"],
                #"LDA": categorized_lda[article_id]["scores"],
            }
        }

    # Save JSON to file
    filename = os.path.join(feeds_dir, "similarity_results_combined.json")
    with open(filename, "w") as json_file:
        json.dump(combined_data, json_file, indent=4, default=str)
    logger.info(f"✅ Saved: {filename}")

def group_articles():
    load_article_cache()
    # Preprocess articles
    logger.info('Preprocessing')
    preprocessed_articles = [preprocess(article) for article in article_contents]
    # Compute TF-IDF & BoW vectors
    logger.info('TFIDF')
    tfidf_matrix = get_tfidf_matrix(preprocessed_articles)
    logger.info('BOW')
    bow_matrix = get_bow_matrix(preprocessed_articles)

    # Prepare corpus and train LDA model
    logger.info('LDA Training')
    dictionary, corpus = prepare_lda_corpus(preprocessed_articles)
    lda_model = train_lda_model(corpus, dictionary)

    # Get topic vectors & compute similarity
    logger.info('Topic vectors')
    topic_vectors = get_topic_vectors(lda_model, corpus)
    logger.info('LDA similarity')
    lda_similarity = compute_lda_similarity(topic_vectors)

    # Calculate similarity
    logger.info('TFIDF Similarity')
    tfidf_similarity = get_similarity(tfidf_matrix)
    logger.info('BOW Similarity')
    bow_similarity = get_similarity(bow_matrix)
    logger.info('Jaccard Similarity')
    jaccard_similarity = get_jaccard_similarity(preprocessed_articles)
    if use_gpu:
        logger.info('LSA Similarity (with GPU)')
        lsa_similarity = get_lsa_similarity_gpu(tfidf_matrix)
    else:
        logger.info('LSA Similarity')
        lsa_similarity = get_lsa_similarity(tfidf_matrix)

    # Find most similar articles for each entry
    logger.info('Finding most similar articles using TFIDF')
    most_similar_tfidf = find_most_similar(tfidf_similarity)
    logger.info('Finding most similar articles using BOW')
    most_similar_bow = find_most_similar(bow_similarity)
    logger.info('Finding most similar articles using Jaccard')
    most_similar_jaccard = find_most_similar(jaccard_similarity)
    logger.info('Finding most similar articles using Latent Semantic Analysis')
    most_similar_lsa = find_most_similar(lsa_similarity)

    # Categorize LDA similarities
    logger.info('Finding most similar articles using Latent Dirichlet Allocation')
    categorized_lda = categorize_similarity(lda_similarity, strong_threshold=0.75, medium_threshold=0.50)


    # Categorize similarity scores for different methods
    categorized_tfidf = categorize_similarity(tfidf_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_bow = categorize_similarity(bow_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_jaccard = categorize_similarity(jaccard_similarity, strong_threshold=0.50, medium_threshold=0.30)
    categorized_lsa = categorize_similarity(lsa_similarity, strong_threshold=0.75, medium_threshold=0.50)

    # Save combined JSON with all similarity methods
    save_combined_json(categorized_tfidf, categorized_bow, categorized_jaccard, categorized_lsa, categorized_lda)


### **Main Execution**
if __name__ == "__main__":

    load_article_cache()
    
    # Preprocess articles
    preprocessed_articles = [preprocess(article) for article in article_contents]

    # Compute TF-IDF & BoW vectors
    tfidf_matrix = get_tfidf_matrix(preprocessed_articles)
    bow_matrix = get_bow_matrix(preprocessed_articles)

    # Prepare corpus and train LDA model
    dictionary, corpus = prepare_lda_corpus(preprocessed_articles)
    lda_model = train_lda_model(corpus, dictionary)

    # Get topic vectors & compute similarity
    topic_vectors = get_topic_vectors(lda_model, corpus)
    lda_similarity = compute_lda_similarity(topic_vectors)

    # Calculate similarity
    tfidf_similarity = get_similarity(tfidf_matrix)
    bow_similarity = get_similarity(bow_matrix)
    jaccard_similarity = get_jaccard_similarity(preprocessed_articles)
    lsa_similarity = get_lsa_similarity(tfidf_matrix)

    # Find most similar articles for each entry
    logger.info('Finding most similar articles using TFIDF')
    most_similar_tfidf = find_most_similar(tfidf_similarity)
    logger.info('Finding most similar articles using BOW')
    most_similar_bow = find_most_similar(bow_similarity)
    logger.info('Finding most similar articles using Jaccard')
    most_similar_jaccard = find_most_similar(jaccard_similarity)
    logger.info('Finding most similar articles using Latent Semantic Analysis')
    most_similar_lsa = find_most_similar(lsa_similarity)

    # Categorize LDA similarities
    logger.info('Finding most similar articles using Latent Dirichlet Allocation')
    categorized_lda = categorize_similarity(lda_similarity, strong_threshold=0.75, medium_threshold=0.50)


    # Categorize similarity scores for different methods
    categorized_tfidf = categorize_similarity(tfidf_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_bow = categorize_similarity(bow_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_jaccard = categorize_similarity(jaccard_similarity, strong_threshold=0.50, medium_threshold=0.30)
    categorized_lsa = categorize_similarity(lsa_similarity, strong_threshold=0.75, medium_threshold=0.50)

    # Save combined JSON with all similarity methods
    save_combined_json(categorized_tfidf, categorized_bow, categorized_jaccard, categorized_lsa, categorized_lda)

    # Save results with similarity scores
    # save_similarity_to_csv(most_similar_tfidf, tfidf_similarity, "TF-IDF")
    # save_similarity_to_csv(most_similar_bow, bow_similarity, "Bag-of-Words")
    # save_similarity_to_csv(most_similar_jaccard, jaccard_similarity, "Jaccard")
    # save_similarity_to_csv(most_similar_lsa, lsa_similarity, "LSA")

    # Generate and save heatmaps
    # logger.info('Generating TFIDF Heatmap')
    # plot_heatmap(tfidf_similarity, "TF-IDF")
    # logger.info('Generating BOW Heatmap')
    # plot_heatmap(bow_similarity, "Bag-of-Words")
    # plot_heatmap(jaccard_similarity, "Jaccard Similarity")
    # plot_heatmap(lsa_similarity, "LSA")

    # # Display Results
    # tfidf_similarity_df = pd.DataFrame(tfidf_similarity, 
    #                              columns=["Article 1", "Article 2"], 
    #                              index=["Article 1", "Article 2"])
    # logger.info("\nText Similarity Matrix:")
    # logger.info(tfidf_similarity_df)

    # bow_similarity_df = pd.DataFrame(bow_similarity, 
    #                              columns=["Article 1", "Article 2"], 
    #                              index=["Article 1", "Article 2"])
    # logger.info("\nBag-of-Words Similarity Matrix:")
    # logger.info(bow_similarity_df)

    # similarity_df = pd.DataFrame({
    #     "TF-IDF Similarity": [tfidf_similarity[0,1]],
    #     "Bag-of-Words Similarity": [bow_similarity[0,1]]
    # })

    # logger.info("\nText Similarity Scores:")
    # logger.info(similarity_df)
