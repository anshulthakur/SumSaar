import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel

from settings import PROJECT_DIRS
feeds_dir = PROJECT_DIRS.get('runtime')

# Ensure required nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load spaCy model for NER and event extraction
nlp = spacy.load("en_core_web_sm")

# === Load Articles ===
with open(os.path.join(feeds_dir, 'cache.json'), 'r') as fd:
    cache = json.load(fd)

# Sort articles by ID and store (id, title, content)
articles = sorted(cache, key=lambda x: x['id'])
article_contents = [article['content'] for article in articles]

# Create mapping of index to (id, title)
index_to_metadata = {i: (article['id'], article['title']) for i, article in enumerate(articles)}

# === Preprocessing Function ===
def preprocess(text):
    """Tokenize, remove stopwords, apply stemming & lemmatization"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return " ".join(lemmatized_tokens)

# === Vector Representations & Similarity Functions ===

def get_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus)

def get_bow_matrix(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

def get_similarity(matrix):
    return cosine_similarity(matrix)

def get_jaccard_similarity(corpus):
    num_articles = len(corpus)
    jaccard_matrix = np.zeros((num_articles, num_articles))
    for i in range(num_articles):
        set_i = set(corpus[i].split())
        for j in range(num_articles):
            set_j = set(corpus[j].split())
            union = set_i.union(set_j)
            jaccard_matrix[i, j] = len(set_i & set_j) / len(union) if union else 0
    return jaccard_matrix

def get_lsa_similarity(tfidf_matrix, n_components=100):
    svd = TruncatedSVD(n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    return cosine_similarity(lsa_matrix)

# === LDA Functions ===

def prepare_lda_corpus(texts):
    """Convert preprocessed texts into a dictionary and BOW corpus."""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return dictionary, corpus

def train_lda_model(corpus, dictionary, num_topics=5):
    """Train an LDA model to extract topics."""
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model

def get_topic_vectors(lda_model, corpus):
    """Convert articles into topic distribution vectors."""
    topic_vectors = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_vector = np.array([prob for _, prob in topics])
        topic_vectors.append(topic_vector)
    return np.array(topic_vectors)

def compute_lda_similarity(topic_vectors):
    return cosine_similarity(topic_vectors)

# === Heatmap Plotting ===

def plot_heatmap(similarity_matrix, method_name):
    plt.figure(figsize=(15, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title(f"Article Similarity Heatmap ({method_name})")
    plt.savefig(os.path.join(feeds_dir, f"similarity_heatmap_{method_name}.png"))
    # plt.show()

# === Find Most Similar Articles (for reference) ===

def find_most_similar(similarity_matrix):
    most_similar = {}
    num_articles = len(article_contents)
    for i in range(num_articles):
        sims = similarity_matrix[i]
        most_similar_index = np.argsort(sims)[-2]  # skip itself (max value)
        most_similar[index_to_metadata[i]] = index_to_metadata[most_similar_index]
    return most_similar

# === Categorize Similarity Scores Based on Thresholds ===

def categorize_similarity(similarity_matrix, strong_threshold, medium_threshold):
    """Categorizes similar articles (based on a given similarity matrix)
    using the global mapping index_to_metadata.
    Returns a dict keyed by article id, with groups for 'strong' and 'medium'."""
    categorized_data = {}
    num_articles = len(article_contents)
    for i in range(num_articles):
        id1, title1 = index_to_metadata[i]
        categorized_data[id1] = {
            "id": id1,
            "title": title1,
            "scores": {"strong": [], "medium": []}
        }
        for j in range(num_articles):
            if i == j:
                continue
            id2, title2 = index_to_metadata[j]
            sim_score = round(similarity_matrix[i, j], 4)
            if sim_score >= strong_threshold:
                categorized_data[id1]["scores"]["strong"].append({"id": id2, "title": title2, "similarity": sim_score})
            elif sim_score >= medium_threshold:
                categorized_data[id1]["scores"]["medium"].append({"id": id2, "title": title2, "similarity": sim_score})
    return categorized_data

# === spaCy-Based Topic Verification Functions ===

def extract_entities(text):
    """Extract and return a set of key entities from the text using spaCy."""
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in {"GPE", "DATE", "EVENT", "LOC", "ORG", "PERSON"}:
            entities.add(ent.text.lower())
    return entities

def compute_topic_similarity_matrix_spacy(texts):
    """Compute a topic similarity (Jaccard) matrix based on named entities extracted by spaCy."""
    num_articles = len(texts)
    topic_mat = np.zeros((num_articles, num_articles))
    entities_list = []
    for text in texts:
        ents = extract_entities(text)
        entities_list.append(ents)
    for i in range(num_articles):
        for j in range(num_articles):
            union = entities_list[i].union(entities_list[j])
            if union:
                topic_mat[i, j] = len(entities_list[i].intersection(entities_list[j])) / len(union)
            else:
                topic_mat[i, j] = 0
    return topic_mat

def get_topic_verification_status(topic_matrix, topic_threshold=0.3):
    """For each article, returns True if it has at least one other article
    with topic similarity above the threshold."""
    topic_status = {}
    n = topic_matrix.shape[0]
    for i in range(n):
        verified = any(topic_matrix[i, j] >= topic_threshold for j in range(n) if j != i)
        aid, _ = index_to_metadata[i]
        topic_status[aid] = verified
    return topic_status

# === Content Coverage Analysis Functions ===

def analyze_content_coverage(text1, text2, subset_threshold=0.8):
    """
    Compare two texts (using token sets) to decide their relationship.
    Computes the proportion of tokens in each text that are shared.
    Returns:
      relation: "subset" if text1 is mostly contained in text2,
                "superset" if text1 largely contains text2,
                "intersection" otherwise.
      unique_info: list of tokens that are unique differences (or shared tokens for intersection).
      coverage1, coverage2: fraction of tokens in text1 and text2 that are in the intersection.
    """
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 or not tokens2:
        return "intersection", [], 0, 0
    intersection = tokens1.intersection(tokens2)
    coverage1 = len(intersection) / len(tokens1)
    coverage2 = len(intersection) / len(tokens2)
    if coverage1 >= subset_threshold and coverage1 < 1.0:
        relation = "subset"  # text1 is largely contained in text2
        unique_info = list(tokens2 - tokens1)
    elif coverage2 >= subset_threshold and coverage2 < 1.0:
        relation = "superset"  # text1 largely covers text2 (i.e. text2 is subset of text1)
        unique_info = list(tokens1 - tokens2)
    else:
        relation = "intersection"
        unique_info = list(intersection)
    return relation, unique_info, coverage1, coverage2

def analyze_all_content_coverage(texts, topic_matrix, topic_threshold=0.3, subset_threshold=0.8):
    """
    For each pair of articles that are verified to be on the same topic (topic_matrix ≥ threshold),
    analyze their content coverage.
    Returns a dictionary keyed by article id with keys 'subset', 'superset' and 'intersection'.
    """
    n = len(texts)
    content_analysis = { index_to_metadata[i][0]: {"subset": [], "superset": [], "intersection": []} for i in range(n) }
    # Compare each unique pair (i, j) where i < j
    for i in range(n):
        for j in range(i+1, n):
            if topic_matrix[i, j] >= topic_threshold:
                relation, unique_info, cov1, cov2 = analyze_content_coverage(texts[i], texts[j], subset_threshold)
                aid_i, title_i = index_to_metadata[i]
                aid_j, title_j = index_to_metadata[j]
                if relation == "subset":
                    # Article i is a subset of article j
                    content_analysis[aid_i]["subset"].append({"id": aid_j, "title": title_j, "unique_facts": unique_info})
                    content_analysis[aid_j]["superset"].append({"id": aid_i, "title": title_i, "unique_facts": unique_info})
                elif relation == "superset":
                    # Article i is a superset of article j
                    content_analysis[aid_i]["superset"].append({"id": aid_j, "title": title_j, "unique_facts": unique_info})
                    content_analysis[aid_j]["subset"].append({"id": aid_i, "title": title_i, "unique_facts": unique_info})
                elif relation == "intersection":
                    content_analysis[aid_i]["intersection"].append({"id": aid_j, "title": title_j, "shared_facts": unique_info})
                    content_analysis[aid_j]["intersection"].append({"id": aid_i, "title": title_i, "shared_facts": unique_info})
    return content_analysis

# === Save Combined JSON Output ===

def save_combined_json(cat_tfidf, cat_bow, cat_jaccard, cat_lsa, cat_lda, topic_status, content_analysis):
    combined_data = {}
    for article_id in cat_tfidf:
        combined_data[article_id] = {
            "id": article_id,
            "title": cat_tfidf[article_id]["title"],
            "topic_verified": topic_status.get(article_id, False),
            "content_analysis": content_analysis.get(article_id, {"subset": [], "superset": [], "intersection": []}),
            "scores": {
                "cosine": cat_tfidf[article_id]["scores"],
                "bag-of-words": cat_bow[article_id]["scores"],
                "Jaccard": cat_jaccard[article_id]["scores"],
                "LSA": cat_lsa[article_id]["scores"],
                "LDA": cat_lda[article_id]["scores"]
            }
        }
    filename = os.path.join(feeds_dir, "similarity_results_combined.json")
    with open(filename, "w") as json_file:
        json.dump(combined_data, json_file, indent=4, default=str)
    print(f"✅ Saved: {filename}")

# === Main Execution ===

if __name__ == "__main__":
    # Preprocess articles
    preprocessed_articles = [preprocess(article) for article in article_contents]

    # Compute TF-IDF & BoW vectors
    tfidf_matrix = get_tfidf_matrix(preprocessed_articles)
    bow_matrix = get_bow_matrix(preprocessed_articles)

    # Prepare corpus and train LDA model
    dictionary, corpus = prepare_lda_corpus(preprocessed_articles)
    lda_model = train_lda_model(corpus, dictionary)
    
    # Get topic vectors & compute LDA similarity
    topic_vectors = get_topic_vectors(lda_model, corpus)
    lda_similarity = compute_lda_similarity(topic_vectors)

    # Calculate other similarity matrices
    tfidf_similarity = get_similarity(tfidf_matrix)
    bow_similarity = get_similarity(bow_matrix)
    jaccard_similarity = get_jaccard_similarity(preprocessed_articles)
    lsa_similarity = get_lsa_similarity(tfidf_matrix)

    # Plot heatmaps for the various methods (saved as images)
    # plot_heatmap(tfidf_similarity, "TF-IDF")
    # plot_heatmap(bow_similarity, "Bag-of-Words")
    # plot_heatmap(jaccard_similarity, "Jaccard")
    # plot_heatmap(lsa_similarity, "LSA")
    # plot_heatmap(lda_similarity, "LDA")

    # Categorize similarity scores for each method (using chosen thresholds)
    categorized_tfidf = categorize_similarity(tfidf_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_bow = categorize_similarity(bow_similarity, strong_threshold=0.85, medium_threshold=0.65)
    categorized_jaccard = categorize_similarity(jaccard_similarity, strong_threshold=0.50, medium_threshold=0.30)
    categorized_lsa = categorize_similarity(lsa_similarity, strong_threshold=0.75, medium_threshold=0.50)
    categorized_lda = categorize_similarity(lda_similarity, strong_threshold=0.75, medium_threshold=0.50)

    # ===== Topic Verification using spaCy =====
    topic_sim_matrix = compute_topic_similarity_matrix_spacy(article_contents)
    #plot_heatmap(topic_sim_matrix, "Topic_Similarity_SpaCy")
    topic_status = get_topic_verification_status(topic_sim_matrix, topic_threshold=0.3)

    # ===== Content Coverage Analysis =====
    content_analysis = analyze_all_content_coverage(preprocessed_articles, topic_sim_matrix, topic_threshold=0.3, subset_threshold=0.8)

    # Save the combined JSON with all information
    save_combined_json(categorized_tfidf, categorized_bow, categorized_jaccard, categorized_lsa, categorized_lda, topic_status, content_analysis)
