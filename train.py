import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import os
import spotipy
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import pickle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from kneed import KneeLocator
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances, silhouette_score
from scipy.spatial.distance import cdist
from yellowbrick.target import FeatureCorrelation
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

warnings.filterwarnings("ignore")

class ArtistClusterAnalyzer:
    def __init__(self, data):
        self.data = data
        self.X = data.select_dtypes(np.number)
        self.scaler = RobustScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def find_optimal_clusters(self, max_clusters=30):
        inertias = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, kmeans.labels_))

        kn = KneeLocator(
            range(2, max_clusters + 1), inertias,
            curve='convex', direction='decreasing'
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(2, max_clusters + 1)),
            y=inertias,
            name='Inertia'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(2, max_clusters + 1)),
            y=silhouette_scores,
            name='Silhouette Score',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Elbow Method & Silhouette Analysis',
            xaxis_title='Number of Clusters',
            yaxis_title='Inertia',
            yaxis2=dict(
                title='Silhouette Score',
                overlaying='y',
                side='right'
            )
        )

        return kn.elbow, fig

    def perform_kmeans_clustering(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        return self.cluster_labels

    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = self.dbscan.fit_predict(self.X_scaled)
        return self.dbscan_labels

    def create_tsne_projection(self, perplexity=40, n_iter=1000):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            learning_rate='auto',
            init='pca'
        )

        self.embedding = tsne.fit_transform(self.X_scaled)
        return self.embedding

    def visualize_clusters(self, labels, method='kmeans'):
        projection = pd.DataFrame({
            'x': self.embedding[:, 0],
            'y': self.embedding[:, 1],
            'name': self.data['artists'],
            'cluster': labels
        })

        cluster_centers = pd.DataFrame({
            'x': [self.embedding[labels == i, 0].mean() for i in range(labels.max() + 1)],
            'y': [self.embedding[labels == i, 1].mean() for i in range(labels.max() + 1)]
        })

        fig = px.scatter(
            projection,
            x='x',
            y='y',
            color='cluster',
            hover_data=['name'],
            title=f'Artist Clusters ({method.upper()})'
        )

        for i in range(len(cluster_centers)):
            fig.add_trace(
                go.Scatter(
                    x=[cluster_centers.loc[i, 'x']],
                    y=[cluster_centers.loc[i, 'y']],
                    mode='markers+text',
                    marker=dict(symbol='x', size=15, color='black'),
                    text=[f'Cluster {i}'],
                    name=f'Center {i}'
                )
            )

        fig.update_layout(
            template='plotly_white',
            legend_title_text='Cluster',
            hovermode='closest'
        )

        return fig

def analyze_artist_clusters(artist_data, max_clusters=30):
    analyzer = ArtistClusterAnalyzer(artist_data)
    optimal_clusters, elbow_fig = analyzer.find_optimal_clusters(max_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")

    kmeans_labels = analyzer.perform_kmeans_clustering(optimal_clusters)
    dbscan_labels = analyzer.perform_dbscan_clustering()

    embedding = analyzer.create_tsne_projection()

    kmeans_fig = analyzer.visualize_clusters(kmeans_labels, 'kmeans')
    dbscan_fig = analyzer.visualize_clusters(dbscan_labels, 'dbscan')

    return {
        'analyzer': analyzer,
        'optimal_clusters': optimal_clusters,
        'elbow_plot': elbow_fig,
        'kmeans_plot': kmeans_fig,
        'dbscan_plot': dbscan_fig
    }

def analyze_and_save_results(artist_data, max_clusters=30):
    results = analyze_artist_clusters(artist_data)
    analyzer = results['analyzer']

    # Save the results and analyzer
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('analyzer.pkl', 'wb') as f:
        pickle.dump(analyzer, f)

    return results, analyzer

def load_results_and_analyzer():
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('analyzer.pkl', 'rb') as f:
        analyzer = pickle.load(f)
    return results, analyzer

def train_and_save_model(artist_data, analyzer):
    X = analyzer.X_scaled
    y = artist_data['artists']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model and label encoder
    model.save('artist_recommender_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, label_encoder

def load_model_and_label_encoder():
    model = keras.models.load_model('artist_recommender_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def recommend_artists_nn(artist_name, artist_data, analyzer, model, label_encoder, n_artists=5):
    try:
        artist_features = artist_data[artist_data['artists'] == artist_name].select_dtypes(np.number)
        if artist_features.empty:
            print(f"Artist '{artist_name}' not found in the dataset.")
            return []

        scaled_features = analyzer.scaler.transform(artist_features)
        predictions = model.predict(scaled_features)
        top_indices = np.argsort(predictions[0])[::-1][1:n_artists+1]

        recommendations = [label_encoder.inverse_transform([i])[0] for i in top_indices]
        return recommendations

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def main():
    artist_data = pd.read_csv('data/data_by_artist.csv')

    # Load the results and analyzer
    results, analyzer = load_results_and_analyzer()

    # Load the model and label encoder
    model, label_encoder = load_model_and_label_encoder()

    artist_name = input('Enter the artist name: ')
    recommendations = recommend_artists_nn(artist_name, artist_data, analyzer, model, label_encoder)
    print(f"Recommended artists for '{artist_name}': {recommendations}")

if __name__ == "__main__":
    main()
