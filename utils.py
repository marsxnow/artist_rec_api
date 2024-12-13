import logging
import pickle
import pandas as pd
from model import ArtistClusterAnalyzer
from train import load_model_and_label_encoder

def load_resources():
    global ARTIST_DATA, ANALYZER, MODEL, LABEL_ENCODER
    try:
        logging.debug("Loading artist data")
        ARTIST_DATA = pd.read_csv('data/data_by_artist.csv')

        logging.debug("Loading analyzer")
        ANALYZER = ArtistClusterAnalyzer(ARTIST_DATA)

        logging.debug("Loading model and label encoder")
        MODEL, LABEL_ENCODER = load_model_and_label_encoder()

        logging.info("Resources loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        # Attempt to recreate the analyzer if loading fails
        if ARTIST_DATA is not None:
            logging.debug("Recreating analyzer")
            ANALYZER = ArtistClusterAnalyzer(ARTIST_DATA)
