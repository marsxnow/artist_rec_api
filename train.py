# importing the necessary libraries
import numpy as np
import pandas as pd
import plotly.express as px
import warnings
import pickle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy.spatial.distance import cdist
from model import ArtistClusterAnalyzer

warnings.filterwarnings("ignore")
# Define the function to analyze artist clusters
# This function will perform clustering on the artist data
# and return the results and visualizations

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
    with open('models/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('models/analyzer.pkl', 'wb') as f:
        pickle.dump(analyzer, f)

    return results, analyzer

def load_results_and_analyzer():
    with open('models/results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('models/analyzer.pkl', 'rb') as f:
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
    model.save('models/artist_recommender_model.h5')
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, label_encoder

def load_model_and_label_encoder():
    model = keras.models.load_model('models/artist_recommender_model.h5')
    with open('models/label_encoder.pkl', 'rb') as f:
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
