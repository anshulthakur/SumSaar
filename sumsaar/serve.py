import os
import json
import streamlit as st

from settings import PROJECT_DIRS
feeds_dir = PROJECT_DIRS.get('runtime')

def display():
    
    articles = []
    try:
        with open(os.path.join(feeds_dir, 'rewritten_articles.json'), "r") as json_file:
            articles = json.load(json_file)
    except:
        pass

    # Streamlit UI
    st.title("Article Viewer")

    # Convert articles into a dictionary
    article_titles = [article["title"] for article in articles]
    selected_title = st.selectbox("Select an article:", article_titles)

    # Find selected article
    selected_article = next((article for article in articles if article["title"] == selected_title), None)

    # Display article content
    if selected_article:
        st.markdown(selected_article["content"], unsafe_allow_html=True)


if __name__=="__main__":
    display()