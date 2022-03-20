from PyTsetlinMachineCUDA.tm import AETsetlinMachine, TsetlinMachine
import torchvision.transforms as PyTorchTransform
from sklearn.metrics import mean_squared_error
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


#size of batch
k = 1000






mse_train = []
mse_test = []


nb_clauses = [28*28*10, 28*28*100, 28*28*500, 28*28*1000]
T_value = [int(nb_clauses//2), nb_clauses, nb_clauses*2]
s_value = [2.0, 5.0, 10.0, 20.0]


#define parameters for TM Autoencoder 
nb_clauses = 28*28*100
T_value = nb_clauses*2
s_value = 2.0
ae_tm = AETsetlinMachine(int(nb_clauses), T_value, s_value, number_of_classes=28*28, feature_drop_p=0.0)
#training autoencoder
for i in tqdm(range(100)):
    mse_list_train = []
    for x_indx in tqdm(range(0, X_train.shape[0], k), disable=True):
        batch_of_images = X_train[x_indx:x_indx+k]
        ae_tm.fit(batch_of_images, batch_of_images, epochs=1, incremental=True)
        
        prediction = ae_tm.predict(batch_of_images)
        mse_after_epoch = mean_squared_error(prediction, batch_of_images)
        mse_list_train.append(mse_after_epoch)
    
    mse_result_train = np.mean(mse_list_train).item()
    print('MSE after {}. epoch: {}'.format(i+1, mse_result_train))
    mse_train.append(mse_result_train)


    mse_list_test = []
    for x_indx in tqdm(range(0, X_test.shape[0], k), disable=True):
        batch_of_images = X_test[x_indx:x_indx+k]
        prediction = ae_tm.predict(batch_of_images)
        mse_after_epoch = mean_squared_error(prediction, batch_of_images)
        mse_list_test.append(mse_after_epoch)

    mse_result_test = np.mean(mse_list_test).item()
    mse_test.append(mse_result_test)
    print('MSE for test data after {}. epoch: {}'.format(i+1, mse_result_test))
        
    
    
    
    test_batch = X_test[:10,:28*28]
    
    prediction = ae_tm.predict(test_batch)
    
    im = np.concatenate((test_batch.reshape(10*28,28), prediction.reshape(10*28,28)), axis=1)
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.savefig("mnist_fig_reconstruction.png")
    plt.close()

    tm_embs, _ = ae_tm.transform_2(X_train[:1000,:28*28])

    reducer = umap.UMAP(random_state=42)
    reducer.fit(tm_embs)
    coords = reducer.fit_transform(tm_embs)

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], c=Y_train[:1000], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('TM embeddings for MNIST dataset', fontsize=24)
    plt.savefig("umap_fig.png")
    plt.close()

    #visualisation
    #plt.clf()

    #plt.figure()
    _, axis = plt.subplots(1, 1, figsize=(20, 5))
    axis.set_title("MSE for Tsetlin Machine, {} clauses, T = {}, s = {}".format(nb_clauses, T_value, s_value))
    axis.set_xlabel("Epochs")
    axis.set_ylabel("MSE")

    axis.plot(
        list(range(len(mse_train))), mse_train, "o-", color="r", label="Training MSE"
    )
    axis.plot(
        list(range(len(mse_test))), mse_test, "o-", color="b", label="Test MSE"
    )
    axis.legend(loc="best")

    plt.savefig("mse_curve.png")
    plt.close()