import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def preprocess_data(data, required_columns):
    if not all(column in data.columns for column in required_columns):
        print(f"The dataset must contain the following columns: {required_columns}")
        return None
    data = data.dropna(subset=required_columns)
    features = data[required_columns]
    return features

def standardize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def determine_optimal_clusters(scaled_features):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow_method(wcss):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def apply_kmeans(scaled_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters

def visualize_clusters(data, features, cluster_column):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data, x=features[0], y=features[1], hue=cluster_column, palette='viridis', s=100)
    plt.title('Customer Segments')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend(title='Cluster')
    plt.show()

def main():
    file_path = 'Mall_Customers.csv'
    required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']

    data = load_data(file_path)
    if data is None:
        return
    
    features = preprocess_data(data, required_columns)
    if features is None:
        return
    
    scaled_features = standardize_features(features)
    
    wcss = determine_optimal_clusters(scaled_features)
    
    plot_elbow_method(wcss)
    
    optimal_clusters = 5  
    data['Cluster'] = apply_kmeans(scaled_features, optimal_clusters)
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    cluster_means = data.groupby('Cluster')[numeric_columns].mean()
    print(cluster_means)
    
    visualize_clusters(data, ['Annual Income (k$)', 'Spending Score (1-100)'], 'Cluster')

if __name__== "__main__":
    main()