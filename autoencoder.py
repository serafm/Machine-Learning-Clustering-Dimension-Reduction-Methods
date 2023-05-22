import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense


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


def auto_encoder():

    # Load train dataset
    data = pd.read_csv("data/train.csv")

    # Split the dataset into features (X) and target (y)
    X = data.iloc[:, :-1]
    true_labels = data['price_range']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    M = [2, 10, 50]

    for m in M:
        # Define the architecture
        input_dim = 20  # Input dimension
        encoding_dim = m  # Dimension of the encoded representation

        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder_layer1 = Dense(100, activation='relu')(input_layer)
        encoder_layer2 = Dense(encoding_dim, activation='relu')(encoder_layer1)

        # Decoder
        decoder_layer1 = Dense(100, activation='relu')(encoder_layer2)
        decoder_layer2 = Dense(input_dim, activation='sigmoid')(decoder_layer1)

        # Create the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder_layer2)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the autoencoder
        autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32)

        # Create the encoder model by accessing the encoder layers from the trained model
        encoder_model = Model(inputs=autoencoder.input, outputs=encoder_layer2)

        # Pass the data (e.g., X_train_scaled) through the encoder model to obtain the encoded data
        encoded_data = encoder_model.predict(X_scaled)

        scaler = StandardScaler()
        encoded_data_scaled = scaler.fit_transform(encoded_data)

        print("Autoencoder with architecture 20-100-M-100-20 where M=", m)

        K = [2, 4, 6, 8, 10]

        # Perform k-means clustering
        for k in K:
            purity = []
            f_measure = []
            for i in range(10):

                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(encoded_data_scaled)

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

        # Perform agglomerative hierarchical clustering
        for k in K:
            purity = []
            f_measure = []
            for i in range(10):
                agglomerative_clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
                agglomerative_clustering.fit(encoded_data_scaled)

                labels = agglomerative_clustering.labels_

                # Calculate Purity
                purity.append(round(calculate_purity(labels, true_labels), 2))

                # Calculate F-measure
                f_measure.append(round(calculate_f_measure(labels, true_labels), 2))

            # Calculate Average of purity and f-measure
            purity_mo = round(sum(purity)/len(purity), 2)
            f_measure_mo = round(sum(f_measure)/len(f_measure), 2)

            print("Average Agglomerative Hierarchy with K=", k)
            print("Purity=", purity_mo)
            print("F-measure=", f_measure_mo, "\n")


auto_encoder()

