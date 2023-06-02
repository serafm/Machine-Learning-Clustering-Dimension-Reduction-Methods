import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def calculate_purity(predicted_labels, true_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    return purity


def calculate_f_measure(predicted_labels, true_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    tp = cm.diagonal()
    precision = tp / np.sum(cm, axis=0)
    recall = tp / np.sum(cm, axis=1)
    f_measure = 2 * (precision * recall) / (precision + recall)
    f_measure[np.isnan(f_measure)] = 0  # Replace NaNs with 0 (if any)
    return np.mean(f_measure)


def agglomerative_hierarchy(filename):
    # Load dataset
    dataset = pd.read_csv(filename)

    # Select features for clustering
    X = dataset.iloc[:, :-1]
    true_labels = dataset.iloc[:, -1]

    # Standardize the feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform agglomerative hierarchy clustering
    K = [2, 4, 6, 8, 10]
    for k in K:
        agglomerative_clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
        agglomerative_clustering.fit(X_scaled)

        labels = agglomerative_clustering.labels_

        print("Agglomerative Hierarchy Clustering K=", k)

        # Calculate Purity
        purity = calculate_purity(labels, true_labels)
        print("Purity:", purity)

        # Calculate F-measure
        f_measure = calculate_f_measure(labels, true_labels)
        print("F-measure:", f_measure)

        print()


agglomerative_hierarchy('data/train.csv')
