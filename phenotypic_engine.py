import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # <--- MUST BE HERE

class PhenotypicEngine:
    def _init_(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        # Using n_init='auto' to suppress warnings in newer scikit-learn versions
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

    def analyze_phenotypes(self, df, feature_columns):
        """Scales data, reduces dimensionality, and clusters biological profiles."""
        if df.empty or not feature_columns:
            return df, []
            
        # 1. Isolate the numeric features
        x = df[feature_columns].values
        
        # 2. Standardize the data (mean=0, variance=1)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # 3. Dimensionality Reduction (PCA) for 2D visualization
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=2)
        components = self.pca.fit_transform(x_scaled)
        df['PCA1'] = components[:, 0]
        df['PCA2'] = components[:, 1]
        
        # 4. Unsupervised Clustering (K-Means) on the scaled data
        df['Cluster_ID'] = self.kmeans.fit_predict(x_scaled)
        df['Cluster_ID'] = "Phenotype Profile " + df['Cluster_ID'].astype(str)
        
        explained_var = self.pca.explained_variance_ratio_ * 100
        
        return df, explained_var
