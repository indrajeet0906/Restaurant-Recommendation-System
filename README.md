**Features**

Search restaurants by cuisine and location.

Get personalized recommendations based on user input.

View restaurant details, including name, cuisines, city, average cost for two, and ratings.

**How It Works**

**Data Preprocessing:**

Missing values are handled by filling empty cuisine fields with "Unknown."

Relevant features such as cuisines, cities, and restaurant names are extracted.

**Recommendation Algorithm:**

A TF-IDF Vectorizer encodes textual features.

Cosine Similarity is calculated between user input and restaurant data.

The top N restaurants with the highest similarity scores are recommended.


