from PyTsetlinMachineCUDA.tm import AETsetlinMachine, TsetlinMachine
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import umap.umap_ as umap
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#make images black and white
X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

#define parameters for TM Autoencoder 
nb_clauses = 28*28*2000
ae_tm = AETsetlinMachine(int(nb_clauses), nb_clauses*2, 20.0, number_of_classes=28*28, feature_drop_p=0.0)
#size of batch
k = 1000
#switch on noise
with_noise = True

#training autoencoder
for i in tqdm(range(100), disable=True):
    
    mse_list = []
    for x_indx in tqdm(range(0, X_train.shape[0], k), disable=True):
        batch_of_images = X_train[x_indx:x_indx+k]
        
        input_batch = batch_of_images.copy()
        if with_noise:
            binary_noise = np.random.choice([0, 1], size=batch_of_images.size, p=[0.5,0.5]).reshape(batch_of_images.shape)
            input_batch[binary_noise == 0] = 0
        
        
        ae_tm.fit(input_batch, batch_of_images, epochs=1, incremental=True)

        prediction = ae_tm.predict(batch_of_images)
        mse_after_epoch = mean_squared_error(prediction, batch_of_images)
        mse_list.append(mse_after_epoch)

    print('MSE after {}. epoch: {}'.format(i+1, np.mean(mse_list).item()))
    
    test_batch = X_test[:10,:28*28]
    
    prediction = ae_tm.predict(test_batch)
    test_mse = mean_squared_error(prediction, test_batch)
    print('MSE for test batch after {}. epoch: {}'.format(test_mse, i))


    input_batch_with_noise = test_batch.copy()
    if with_noise:
        binary_noise = np.random.choice([0, 1], size=test_batch.size, p=[0.5,0.5]).reshape(test_batch.shape)
        input_batch_with_noise[binary_noise == 0] = 0
    prediction_with_noise = ae_tm.predict(input_batch_with_noise)
    
    im = np.concatenate((test_batch.reshape(10*28,28), input_batch_with_noise.reshape(10*28,28), prediction_with_noise.reshape(10*28,28), prediction.reshape(10*28,28)), axis=1)
    plt.clf()
    plt.imshow(im, cmap='gray')
    plt.savefig("mnist_fig_reconstruction.png")

    tm_embs, _ = ae_tm.transform_2(X_train[:100,:28*28])

    reducer = umap.UMAP(random_state=42)
    reducer.fit(tm_embs)
    coords = reducer.fit_transform(tm_embs)

    plt.clf()
    plt.scatter(coords[:, 0], coords[:, 1], c=Y_train[:100], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    #plt.title('TM embeddings for MNIST dataset', fontsize=24)
    plt.savefig("TM embeddings for MNIST dataset.png")