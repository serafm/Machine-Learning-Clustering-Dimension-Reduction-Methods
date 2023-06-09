import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
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


def kmeans(filename):
    # Load dataset
    dataset = pd.read_csv(filename)

    # Select features for clustering
    X = dataset.iloc[:, :-1]
    true_labels = dataset.iloc[:, -1]

    # Standardize the feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform k-means clustering
    K = [2, 4, 6, 8, 10]
    for k in K:
        purity = []
        f_measure = []
        for i in range(10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)

            labels = kmeans.labels_

            # Calculate Purity
            purity.append(round(calculate_purity(labels, true_labels), 2))

            # Calculate F-measure
            f_measure.append(round(calculate_f_measure(labels, true_labels), 2))

        # Calculate Average of purity and f-measure
        purity_mo = round(sum(purity) / len(purity), 2)
        f_measure_mo = round(sum(f_measure) / len(f_measure), 2)

        print("Average K-means scores with K=", k)
        print("Purity=", purity_mo)
        print("F-measure=", f_measure_mo, "\n")


kmeans('data/train.csv')
