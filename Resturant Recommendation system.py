# app.py

# Import libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'C:/Users/LENOVO/Downloads/Dataset .csv'  # Adjust to your actual file path
    data = pd.read_csv(file_path)
    data['Cuisines'].fillna('Unknown', inplace=True)
    data['Combined Features'] = data['Cuisines'] + ' ' + data['City']
    return data

# Load data
restaurant_data = load_data()

# Create recommendation function
def recommend_restaurants(preferences, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    feature_matrix = tfidf.fit_transform(restaurant_data['Combined Features'])
    user_vector = tfidf.transform([preferences])
    similarity_scores = cosine_similarity(user_vector, feature_matrix)
    recommendations = similarity_scores.argsort()[0][-top_n:][::-1]

    recommended_restaurants = []
    for idx in recommendations:
        recommended_restaurants.append({
            "Restaurant Name": restaurant_data.iloc[idx]['Restaurant Name'],
            "Cuisines": restaurant_data.iloc[idx]['Cuisines'],
            "City": restaurant_data.iloc[idx]['City'],
            "Average Cost for Two": restaurant_data.iloc[idx]['Average Cost for two'],
            "Rating": restaurant_data.iloc[idx]['Aggregate rating']
        })
    return recommended_restaurants


# Streamlit App Layout
st.title("Restaurant Recommendation System")
st.subheader("Enter your restaurant preferences:")

# User input
user_input = st.text_input("Type cuisine or city (e.g., 'Italian Delhi')", "")

if st.button("Get Recommendations"):
    if user_input:
        results = recommend_restaurants(user_input, top_n=5)
        if results:
            for res in results:
                st.write(f"**Restaurant Name:** {res['Restaurant Name']}")
                st.write(f"**Cuisines:** {res['Cuisines']}")
                st.write(f"**City:** {res['City']}")
                st.write(f"**Average Cost for Two:** {res['Average Cost for Two']}")
                st.write(f"**Rating:** {res['Rating']}")
                st.markdown("---")
        else:
            st.error("No matching restaurants found. Try different preferences.")
    else:
        st.warning("Please enter your preferences.")
