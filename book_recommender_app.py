# =========================================
# 📚 BOOK RECOMMENDER SYSTEM
# =========================================

import os
import random
import pickle
import requests
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender", layout="wide")

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    books_path = os.path.join(base_path, "books_dict.pkl")

    with open(books_path, "rb") as f:
        books_dict = pickle.load(f)

    books = pd.DataFrame(books_dict)

    # Make sure required columns exist
    if "title" not in books.columns:
        raise ValueError("books_dict.pkl must contain a 'title' column.")

    if "description" not in books.columns:
        books["description"] = ""

    books["title"] = books["title"].fillna("").astype(str)
    books["description"] = books["description"].fillna("").astype(str)

    return books


# =========================================
# BUILD SIMILARITY
# =========================================
@st.cache_resource
def build_similarity(books):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(books["description"])
    return cosine_similarity(matrix)


# =========================================
# GOOGLE BOOKS API
# =========================================
def fetch_book_details(book_title):
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={book_title}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            volume_info = data["items"][0].get("volumeInfo", {})

            thumbnail = volume_info.get("imageLinks", {}).get("thumbnail", None)
            description = volume_info.get("description", "No description available")
            authors = ", ".join(volume_info.get("authors", ["Unknown"]))
            rating = volume_info.get("averageRating", "N/A")

            return thumbnail, description[:220], authors, rating

    except Exception:
        pass

    return None, "No description available", "Unknown", "N/A"


# =========================================
# RECOMMEND FUNCTION
# =========================================
def recommend(book_title, books, similarity):
    match = books[books["title"] == book_title]

    if match.empty:
        return [], [], [], [], []

    index = match.index[0]
    distances = similarity[index]

    book_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names, posters, descriptions, authors_list, ratings = [], [], [], [], []

    for item in book_list:
        idx = item[0]
        title = books.iloc[idx]["title"]

        thumb, desc, author, rating = fetch_book_details(title)

        names.append(title)
        posters.append(thumb)
        descriptions.append(desc)
        authors_list.append(author)
        ratings.append(rating)

    return names, posters, descriptions, authors_list, ratings


# =========================================
# LOAD EVERYTHING
# =========================================
books = load_data()
similarity = build_similarity(books)


# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b1020;
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 95rem;
}

.hero {
    position: relative;
    height: 420px;
    border-radius: 22px;
    overflow: hidden;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

.overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.68) 38%, rgba(0,0,0,0.15) 100%);
    z-index: 1;
}

.faded {
    position: absolute;
    top: 20px;
    left: 34px;
    font-size: 78px;
    line-height: 0.92;
    font-weight: 900;
    color: rgba(255,255,255,0.12);
    letter-spacing: 3px;
    z-index: 2;
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 3;
    padding: 135px 38px 40px 38px;
    max-width: 620px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 12px;
    color: white;
}

.hero-desc {
    font-size: 18px;
    line-height: 1.6;
    color: #e5e7eb;
}

.section-heading {
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
    margin-bottom: 14px;
    color: white;
}

.card-box {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    min-height: 100%;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# HERO SECTION
# =========================================
random_book = random.choice(books["title"].values)
hero_thumb, hero_desc, hero_author, hero_rating = fetch_book_details(random_book)

st.markdown(f"""
<div class="hero">
    <div class="overlay"></div>
    <div class="faded">BOOK<br>RECOMMENDER</div>
    <div class="hero-content">
        <div class="hero-title">{random_book}</div>
        <div class="hero-desc">{hero_desc}</div>
        <p style="margin-top:14px; color:#d1d5db;">✍️ {hero_author} &nbsp; | &nbsp; ⭐ {hero_rating}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================
# SEARCH + BUTTONS
# =========================================
st.markdown('<div class="section-heading">🔍 Select a Book</div>', unsafe_allow_html=True)

selected_book = st.selectbox(
    "Choose a book",
    books["title"].values,
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)

with col1:
    recommend_btn = st.button("📚 Recommend")

with col2:
    surprise_btn = st.button("🎲 Surprise Me")


# =========================================
# SURPRISE ME
# =========================================
if surprise_btn:
    surprise_book = random.choice(books["title"].values)
    st.success(f"Try reading: {surprise_book}")


# =========================================
# RECOMMENDATIONS
# =========================================
if recommend_btn:
    names, posters, descriptions, authors_list, ratings = recommend(selected_book, books, similarity)

    st.markdown('<div class="section-heading">✨ Recommended Books</div>', unsafe_allow_html=True)

    if not names:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)

                if posters[i]:
                    st.image(posters[i], use_container_width=True)
                else:
                    st.info("No cover available")

                st.markdown(f"**{names[i]}**")
                st.write(f"✍️ {authors_list[i]}")
                st.write(f"⭐ {ratings[i]}")
                st.caption(descriptions[i])

                st.markdown('</div>', unsafe_allow_html=True)


# =========================================
# TRENDING SECTION
# =========================================
st.markdown('<div class="section-heading">🔥 Trending Books</div>', unsafe_allow_html=True)

sample_count = min(5, len(books))
trend_books = random.sample(list(books["title"].values), sample_count)
trend_cols = st.columns(sample_count)

for i, book in enumerate(trend_books):
    thumb, desc, author, rating = fetch_book_details(book)

    with trend_cols[i]:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)

        if thumb:
            st.image(thumb, use_container_width=True)

        st.markdown(f"**{book}**")
        st.write(f"⭐ {rating}")

        st.markdown('</div>', unsafe_allow_html=True)
