# =========================================
# 📚 BOOK RECOMMENDER SYSTEM (STREAMLIT)
# =========================================

import streamlit as st
import pickle
import pandas as pd
import requests
import random

st.set_page_config(layout="wide")

# ==============================
# 📚 GOOGLE BOOKS API
# ==============================
def fetch_book_details(book_title):
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={book_title}"
        res = requests.get(url, timeout=5).json()

        if "items" in res:
            book = res["items"][0]["volumeInfo"]

            thumbnail = book.get("imageLinks", {}).get("thumbnail", None)
            description = book.get("description", "No description available")
            authors = ", ".join(book.get("authors", ["Unknown"]))
            rating = book.get("averageRating", "N/A")

            return thumbnail, description[:200], authors, rating

    except:
        pass

    return None, "No description", "Unknown", "N/A"


# ==============================
# 📂 LOAD DATA
# ==============================
@st.cache_data
def load_data():
    books_dict = pickle.load(open("books_dict.pkl", "rb"))
    

    books = pd.DataFrame(books_dict)
    return books, similarity


books = load_data()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def build_similarity(books):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(books["description"].fillna(""))
    return cosine_similarity(matrix)

similarity = build_similarity(books)
# ==============================
# 🤖 RECOMMEND FUNCTION
# ==============================
def recommend(book):
    index = books[books["title"] == book].index[0]
    distances = similarity[index]

    book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names, posters, descs, authors, ratings = [], [], [], [], []

    for i in book_list:
        title = books.iloc[i[0]].title
        thumb, desc, author, rating = fetch_book_details(title)

        names.append(title)
        posters.append(thumb)
        descs.append(desc)
        authors.append(author)
        ratings.append(rating)

    return names, posters, descs, authors, ratings


# ==============================
# 🎨 UI DESIGN
# ==============================
st.markdown("""
<style>
body {
    background-color: #0b0f19;
}

.hero {
    position: relative;
    height: 400px;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 20px;
}

.overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, black, transparent);
}

.faded {
    position: absolute;
    font-size: 80px;
    color: rgba(255,255,255,0.1);
    top: 20px;
    left: 30px;
    font-weight: bold;
}

.hero-content {
    position: absolute;
    bottom: 40px;
    left: 40px;
}

.book-card {
    background: #111827;
    padding: 10px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# 🎲 RANDOM HERO BOOK
# ==============================
random_book = random.choice(books["title"].values)
thumb, desc, author, rating = fetch_book_details(random_book)

st.markdown(f"""
<div class="hero">
    <div class="overlay"></div>
    <div class="faded">BOOK<br>RECOMMENDER</div>
    <div class="hero-content">
        <h1>{random_book}</h1>
        <p>{desc}</p>
        <p>✍️ {author} | ⭐ {rating}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# 🔍 SEARCH
# ==============================
st.subheader("🔍 Select a Book")

selected_book = st.selectbox("", books["title"].values)

col1, col2 = st.columns(2)

with col1:
    recommend_btn = st.button("📚 Recommend")

with col2:
    surprise_btn = st.button("🎲 Surprise Me")


# ==============================
# 🎲 SURPRISE
# ==============================
if surprise_btn:
    st.success(f"Try reading: {random.choice(books['title'].values)}")


# ==============================
# 📚 RECOMMENDATION DISPLAY
# ==============================
if recommend_btn:
    names, posters, descs, authors, ratings = recommend(selected_book)

    st.subheader("✨ Recommended Books")

    cols = st.columns(5)

    for i in range(len(names)):
        with cols[i]:
            st.markdown('<div class="book-card">', unsafe_allow_html=True)

            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("No Image")

            st.markdown(f"**{names[i]}**")
            st.write(f"✍️ {authors[i]}")
            st.write(f"⭐ {ratings[i]}")
            st.caption(descs[i])

            st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# 🔥 TRENDING
# ==============================
st.subheader("🔥 Trending Books")

trend_books = random.sample(list(books["title"].values), 5)
cols = st.columns(5)

for i, book in enumerate(trend_books):
    thumb, _, author, rating = fetch_book_details(book)

    with cols[i]:
        if thumb:
            st.image(thumb)
        st.write(book)
        st.caption(f"⭐ {rating}")
