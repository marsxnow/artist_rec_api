import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from kneed import KneeLocator
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
