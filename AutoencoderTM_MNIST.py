from PyTsetlinMachineCUDA.tm import AETsetlinMachine, TsetlinMachine
import torchvision.transforms as PyTorchTransform
import torch
import numpy as np
from time import time
import cv2
import random
import umap.umap_ as umap

from keras.datasets import mnist

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA



#load MNIST dataset

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#make images black and white
X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

#define parameters for TM Autoencoder 

nb_clauses = 28*28*100
#max_thershold = 20
#ae_tm_1 = AETsetlinMachine(int(nb_clauses), nb_clauses*2, 2.0, number_of_classes=10*10)
ae_tm_2 = AETsetlinMachine(int(nb_clauses), nb_clauses*2, 2.0, number_of_classes=28*28, feature_drop_p=0.0)
#size of batch
k = 1000

#training autoencoder
for i in tqdm(range(100)):
    for x_indx in tqdm(range(0, X_train.shape[0], k), disable=True):
        batch_of_images = X_train[x_indx:x_indx+k]
        ae_tm_2.fit(batch_of_images, batch_of_images, epochs=1, incremental=True)
        
    
    
    
    test_batch = X_test[:10,:28*28]
    
    
    prediction = ae_tm_2.predict(test_batch)
    
    im = np.concatenate((test_batch.reshape(10*28,28), prediction.reshape(10*28,28)), axis=1)
    plt.clf()
    plt.imshow(im, cmap='gray')
    plt.savefig("mnist_fig_reconstruction.png")

    tm_embs, _ = ae_tm_2.transform_2(X_train[:100,:28*28])

    reducer = umap.UMAP(random_state=42)
    reducer.fit(tm_embs)
    coords = reducer.fit_transform(tm_embs)

    plt.clf()
    plt.scatter(coords[:, 0], coords[:, 1], c=Y_train[:100], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('TM embeddings for MNIST dataset', fontsize=24)
    plt.savefig("umap_fig.png")