import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-Commerce Recommender", layout="wide")


@st.cache_data
def load_data():
    data = {
        "product_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "title": [
            "Nike Air Max Shoes",
            "Adidas Ultraboost Running Shoes",
            "Apple iPhone 15 Pro",
            "Samsung Galaxy S24 Ultra",
            "Sony WH-1000XM5 Headphones",
            "Bose QuietComfort 45",
            "Puma Training T-Shirt",
            "Nike Dri-FIT Sports Tee"
        ],
        "category": [
            "Footwear", "Footwear", "Electronics", "Electronics",
            "Audio", "Audio", "Clothing", "Clothing"
        ],
        "description": [
            "Running shoes with air cushioning comfortable sports footwear",
            "Boost foam responsive running footwear for athletics and gym",
            "Flagship smartphone with titanium body A17 pro chip camera",
            "AI powered android flagship smartphone with stylus display",
            "Active noise cancelling wireless over-ear premium headphones",
            "Bluetooth wireless headphones comfortable quiet noise cancelling",
            "Breathable lightweight polyester gym running sports t-shirt",
            "Sweat wicking athletic sports t-shirt for running and training"
        ],
        "price": [120, 140, 999, 1199, 399, 329, 25, 30]
    }
    df = pd.DataFrame(data)
    # Metadata combine karna
    df["features"] = df["title"] + " " + df["category"] + " " + df["description"]
    return df


df = load_data()

# Model Training (TF-IDF + Cosine Similarity)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["features"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(product_name, top_n=3):
    idx = df[df["title"] == product_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected_indices = [i[0] for i in scores[1:top_n + 1]]
    return df.iloc[selected_indices]


# Streamlit UI
st.title("🛒 E-Commerce Smart Recommendation Engine")
st.write("Machine Learning powered product similarity system")

selected_product = st.selectbox("Select a Product to view:", df["title"].values)

col1, col2 = st.columns([1, 2])
current_item = df[df["title"] == selected_product].iloc[0]

with col1:
    st.subheader("Selected Item")
    st.write(f"**Name:** {current_item['title']}")
    st.write(f"**Category:** {current_item['category']}")
    st.write(f"**Price:** ${current_item['price']}")
    st.caption(current_item["description"])

with col2:
    st.subheader("Recommended for You")
    recommendations = recommend(selected_product)

    rec_cols = st.columns(len(recommendations))
    for i, (_, row) in enumerate(recommendations.iterrows()):
        with rec_cols[i]:
            st.markdown(f"**{row['title']}**")
            st.write(f"Category: {row['category']}")
            st.write(f"Price: **${row['price']}**")
            st.caption(row["description"][:60] + "...")