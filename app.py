from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from train import load_model_and_label_encoder, recommend_artists_nn, ArtistClusterAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables to store loaded models and data
ARTIST_DATA = None
ANALYZER = None
MODEL = None
LABEL_ENCODER = None

def load_resources():
    global ARTIST_DATA, ANALYZER, MODEL, LABEL_ENCODER
    try:
        # Load artist data
        ARTIST_DATA = pd.read_csv('data/data_by_artist.csv')

        # Load analyzer
        with open('analyzer.pkl', 'rb') as f:
            ANALYZER = pickle.load(f)

        # Load model and label encoder
        MODEL, LABEL_ENCODER = load_model_and_label_encoder()

        print("Resources loaded successfully!")
    except Exception as e:
        print(f"Error loading resources: {e}")
        # Attempt to recreate the analyzer if loading fails
        if ARTIST_DATA is not None:
            ANALYZER = ArtistClusterAnalyzer(ARTIST_DATA)

@app.route("/")
def hello_world():
    return "<p>Artist Recommendation Service</p>"

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    global ARTIST_DATA, ANALYZER, MODEL, LABEL_ENCODER

    # Check if resources are loaded
    if ARTIST_DATA is None or ANALYZER is None or MODEL is None or LABEL_ENCODER is None:
        load_resources()

    # Get artist name from request
    data = request.json
    artist_name = data.get("artist_name")
    print(f"Received request for artist: {artist_name}")

    if not artist_name:
        return jsonify({"error": "Please provide an artist name"}), 400

    try:
        # Generate recommendations
        recommendations = recommend_artists_nn(
            artist_name,
            ARTIST_DATA,
            ANALYZER,
            MODEL,
            LABEL_ENCODER
        )

        if not recommendations:
            return jsonify({
                "error": f"No recommendations found for artist '{artist_name}'"
            }), 404

        return jsonify({
            "original_artist": artist_name,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

# Load resources when the application starts
load_resources()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
