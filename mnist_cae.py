#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function

import time
import datetime
import os
import shutil
import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder, Dropout, MaxoutDense
from keras.layers.noise import GaussianNoise
from keras.activations import sigmoid
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import noise
import keras.models as models
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, AutoEncoder, Merge
from keras.utils.visualize_util import plot
import keras.callbacks
from keras.regularizers import l2, activity_l2, l1
from keras import backend as K

# Isaac Gerg
# Mimicked from https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.ipynb

# Not my function, I can't recall who this is from.  If this belongs to you, please let me know.
def make_mosaic(imgs, nrows, ncols, border=1):
    import numpy.ma as ma
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i].reshape(imshape)
    return mosaic

#from mpl_toolkits.axes_grid1 import make_axes_locatable


def isSquare(n):
    tmp = np.sqrt(n)
    if np.floor(tmp) == tmp: 
        return True
    else:
        return False

def formatWeights(layer):
    import matplotlib.pyplot as plt
    import matplotlib.cm
    if type(layer) ==  Convolution2D:
        # Get size
        tmp = np.squeeze(layer.get_weights()[0])
        if tmp.ndim == 4:
            tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2], tmp.shape[3])
        if tmp.ndim > 2:
            N = len(tmp)
            rows = int(np.ceil(np.sqrt(N)))
            cols = N/rows + 1
            tmp = make_mosaic(tmp, rows, cols)
        return tmp
    if type(layer) == keras.layers.core.Dense:
        tmp = layer.get_weights()[0]
        
        # Find smallest dimension
        if tmp.shape[0] > tmp.shape[1] and isSquare(tmp.shape[0]):            
            sf0 = np.sqrt(tmp.shape[0])            
            r = np.ceil(np.sqrt(tmp.shape[1]))
            tmp = make_mosaic(np.reshape(np.transpose(tmp), (tmp.shape[1], sf0, sf0)), r, r)                        
        elif tmp.shape[0] > tmp.shape[1] and isSquare(tmp.shape[1]):
            sf1 = np.sqrt(tmp.shape[1])            
            r = np.ceil(np.sqrt(tmp.shape[0]))
            tmp = make_mosaic(np.reshape(tmp, (tmp.shape[0], sf1, sf1)), r, r)              
        # Neither dimension is a perfect square
        else:
            tmp = np.vstack((layer.get_weights()[0], layer.get_weights()[1]))
        return tmp
    
    W = model.get_weights()
    numTiles = W[0].shape[1]
    t = np.sqrt(numTiles)
    numRows = np.ceil(np.sqrt(numTiles))
    numCols = np.ceil(numTiles/numRows)
    m = make_mosaic(W[2], numRows, numCols)
    plt.imshow(m, cmap=matplotlib.cm.gray, interpolation='nearest'); plt.show()    
    return

class LossHistory(keras.callbacks.Callback):
    def __init__(self, outputDir):
        self.outputDir = outputDir
        # Save plots and weights every 1 and 15 minutes respectively.
        self.saveWeightsInterval = datetime.timedelta(minutes=15)
        self.savePlotsInterval = datetime.timedelta(minutes=1)
        self.timeWeightsSaved = datetime.datetime.now()
        self.timePlotsSaved = datetime.datetime.now()
        super(LossHistory, self).__init__()        
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.epochNum = 0
        self.startTime = time.time()
        # Determine 2d conv layers
        self.convLayers = []
        c = 0
        for k in self.model.layers:
            if type(self.model.layers[c]) == keras.layers.convolutional.Convolution2D or type(self.model.layers[c]) == keras.layers.core.Dense:
                self.convLayers.append(c)
            c+=1

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('val_acc'))
        self.epochNum += 1
        
        if datetime.datetime.now() > self.timeWeightsSaved + self.saveWeightsInterval:
            fn = os.path.join(self.outputDir, 'model_weights.h5')
            if os.path.exists(fn):
                os.remove(fn)
            self.model.save_weights(fn, overwrite=True)       
            self.timeWeightsSaved = datetime.datetime.now()
        
        if datetime.datetime.now() > self.timePlotsSaved + self.savePlotsInterval:
            plt.figure()
            plt.subplot(121)
            plt.plot(self.losses)
            plt.title('Loss Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.subplot(122)
            plt.plot(self.acc)
            plt.title('Acc Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Acc')
            plt.ylim(0,1)
            plt.savefig(os.path.join(self.outputDir, 'loss and acc.png'))
            plt.close()
            
            try:
                for k in self.convLayers:
                    plt.figure()
                    w = formatWeights(self.model.layers[k])
                    plt.imshow(w, cmap=matplotlib.cm.viridis) #, vmin=-1, vmax=1)
                    plt.colorbar(); plt.title('Layer %d'%(k,)); 
                    #plt.tight_layout()
                    plt.savefig(os.path.join(self.outputDir, 'Layer %d.png'%(k,)))
                    plt.close()
            except:
                print('Matplotlib error')
                pass
                
            # Print estimated time to finish
            timePerEpoch = (time.time() - self.startTime)/self.epochNum
            epochsRemaining = (self.params['nb_epoch']-self.epochNum)
            timeRemaining = timePerEpoch*epochsRemaining
            etf = (datetime.datetime.now() + datetime.timedelta(seconds=timeRemaining)).isoformat(' ')
            print('-----------------------------------------> Estimated finish time: %s'%(etf,))
            self.timePlotsSaved = datetime.datetime.now()        

def mnistCae():
    #-----------------------------------------------------------------------------------------------
    # Params
    batch_size = 128
    nb_epoch = 10
    conv_num_filters = 16
    filter_size = 3
    pool_size = 2
    encode_size = 16
    dense_mid_size = 128    
    #-----------------------------------------------------------------------------------------------

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
        
    # Build the model
    model = Sequential()
    act = 'sigmoid'
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='valid', activation=act, input_shape=(1, img_rows, img_cols)))
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='valid', activation=act))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Convolution2D(2*conv_num_filters, filter_size, filter_size, border_mode='valid', activation=act))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Flatten())
    model.add(Dense(dense_mid_size, activation=act))
    model.add(Dense(encode_size, activation=act))
    model.add(Dense(dense_mid_size, activation=act))    
    model.add(Dense(800, activation=act))   
    model.add(Reshape((2*conv_num_filters, 5, 5)))
    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same', activation=act))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(UpSampling2D(size=(pool_size, pool_size)))              
    model.add(Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same', activation=act))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(1, filter_size, filter_size, border_mode='same', activation=act))
    model.add(ZeroPadding2D(padding=(1, 1)))
    
    # Show summary of model
    model.summary()
        
    model.compile(loss='mse', optimizer='adadelta')

    # Show graph of model
    outputDir = datetime.datetime.now().strftime("%A, %d. %B %Y %I.%M%p")
    os.mkdir(outputDir)    
    import keras.utils.visualize_util as vutil
    t = vutil.to_graph(model, recursive=True, show_shape=True).create(prog='dot', format="png")
    fid = open(os.path.join(outputDir, 'graph.png'), 'wb'); fid.write(t); fid.close()    
    history = LossHistory(outputDir)
    
    # Train up model
    model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.1, callbacks=[history])
    
    # Show examples of reconstruction
    for k in np.random.randint(0, 1000, size=10):
        img = np.squeeze(X_test[k:k+1,:,:,:])
        reconstruction = np.squeeze(model.predict(X_test[k:k+1,:,:,:]))
        plt.figure()
        plt.subplot(121)
        plt.imshow(img); plt.title('Sample'); plt.colorbar()
        plt.subplot(122)
        plt.imshow(reconstruction); plt.title('Reconstruction'); plt.colorbar()
        plt.savefig(os.path.join(outputDir, 'reconstruction example - %d.png'%k))
         
    return

if __name__ == '__main__':
    mnistCae()
    print('Done.')
