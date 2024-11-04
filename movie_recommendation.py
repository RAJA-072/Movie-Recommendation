import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\raji\Downloads\Movie recommendation\movies_data.csv'
movies_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Step 1: Inspect dataset columns
print("Columns in the dataset:\n", movies_data.columns)

# Step 2: Select relevant features for clustering
features = ['Genre', 'IMDb score']
print("Selected Features: ", features)

# Step 3: Preprocess the Data
# Preprocessing pipeline: OneHotEncoding for 'Genre' and StandardScaler for 'IMDb score'
preprocessor = ColumnTransformer(
    transformers=[
        ('genre', OneHotEncoder(), ['Genre']),
        ('imdb', StandardScaler(), ['IMDb score'])
    ]
)

# Apply the preprocessing to the features
processed_features = preprocessor.fit_transform(movies_data[features])

# Step 4: Perform KMeans Clustering
n_clusters = len(movies_data['Genre'].unique())
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(processed_features)

# Get the initial clusters
initial_clusters = kmeans.labels_
print("Initial Clusters:\n", initial_clusters)

# Step 5: Add the cluster labels to the original dataset
movies_data['Cluster'] = initial_clusters

# Define moods for clusters based on the unique genres found in the data
unique_genres = movies_data['Genre'].unique()
cluster_moods = {i: genre for i, genre in enumerate(unique_genres)}

# List clusters and their actual genres for user selection
print("\nClusters and their genres:")
for cluster_id in range(n_clusters):
    cluster_genres = movies_data[movies_data['Cluster'] == cluster_id]['Genre'].unique()
    print(f"Cluster {cluster_id}: {', '.join(cluster_genres)}")

# QLearningMovieRecommender class definition
class QLearningMovieRecommender:
    def __init__(self, data, cluster, n_recommendations=10):
        self.data = data
        self.cluster = cluster
        self.n_recommendations = n_recommendations
        self.q_table = {}  # Initialize Q-table for state-action pairs
        self.initialize_q_table()

    def initialize_q_table(self):
        """Initialize Q-table for each movie with random Q-values."""
        for movie_id in self.data.index:
            self.q_table[movie_id] = random.uniform(0, 1)  # Random Q-value initialization

    def update_q_value(self, movie_id, reward, learning_rate=0.1, discount_factor=0.95):
        """Update Q-values using the Q-learning formula."""
        old_q_value = self.q_table[movie_id]
        max_next_q = max(self.q_table.values())  # Max future reward
        # Q-learning update rule
        self.q_table[movie_id] = old_q_value + learning_rate * (reward + discount_factor * max_next_q - old_q_value)

    def recommend_movies(self):
        """Generate movie recommendations using Q-learning."""
        recommended_movies = []
        selected_movies = set()

        for _ in range(self.n_recommendations):
            # Select movie with max Q-value in the cluster
            cluster_movies = self.data[self.data['Cluster'] == self.cluster].index
            available_movies = [movie_id for movie_id in cluster_movies if movie_id not in selected_movies]

            if not available_movies:
                break

            best_movie_id = max(available_movies, key=lambda x: self.q_table[x])
            recommended_movies.append(best_movie_id)
            selected_movies.add(best_movie_id)

            # Simulate user feedback (like = +1 reward, dislike = -1 reward)
            feedback = random.choice([1, -1])  # Simulated feedback
            self.update_q_value(best_movie_id, reward=feedback)

        recommendations = self.data.loc[recommended_movies][['Movie', 'Genre', 'IMDb score']]
        return recommendations

# Policy Gradient Algorithm class definition
class PolicyGradientMovieRecommender:
    def __init__(self, data, cluster, n_recommendations=10):
        self.data = data
        self.cluster = cluster
        self.n_recommendations = n_recommendations

    def recommend_movies(self):
        """Generate movie recommendations using a probability-based policy."""
        cluster_movies = self.data[self.data['Cluster'] == self.cluster]
        # Assign random probabilities to each movie in the cluster
        probabilities = np.random.dirichlet(np.ones(len(cluster_movies)), size=1)[0]
        movie_ids = cluster_movies.index

        # Select movies based on probability distribution
        selected_movie_ids = np.random.choice(movie_ids, size=self.n_recommendations, replace=False, p=probabilities)
        recommendations = self.data.loc[selected_movie_ids][['Movie', 'Genre', 'IMDb score']]

        return recommendations

# Example usage
user_input_cluster = int(input(f"Enter cluster number (0 to {n_clusters-1}) to generate recommendations: "))
mood = cluster_moods.get(user_input_cluster, "Unknown Mood")
print(f"Generating recommendations for mood: {mood}")

# Q-learning based recommendation
q_learning_recommender = QLearningMovieRecommender(movies_data, cluster=user_input_cluster)
q_learning_recommendations = q_learning_recommender.recommend_movies()
print(f"\nQ-Learning Recommended Movies for {mood}:\n", q_learning_recommendations)

# Policy Gradient based recommendation
policy_gradient_recommender = PolicyGradientMovieRecommender(movies_data, cluster=user_input_cluster)
policy_gradient_recommendations = policy_gradient_recommender.recommend_movies()
print(f"\nPolicy Gradient Recommended Movies for {mood}:\n", policy_gradient_recommendations)

# Step 6: Visualizing KMeans Clustering
# Reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(processed_features)

# Visualizing KMeans Clustering
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=initial_clusters, cmap='viridis')
plt.title("KMeans Clustering Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.show()
