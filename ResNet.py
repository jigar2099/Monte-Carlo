from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important;}</style>"))
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.utils import shuffle
#sns.set(style='darkgrid')
#from talos.utils import lr_normalizer
import keras
import tensorflow as tf
#from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.text import one_hot, Tokenizer
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense,GRU, Activation, Flatten, Reshape, BatchNormalization, Input, add, GlobalAveragePooling1D, Bidirectional, LSTM, TimeDistributed, RepeatVector#, Attentionn
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, ZeroPadding1D, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling1D
from keras.regularizers import l1, l2
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
RMSE=tf.keras.metrics.RootMeanSquaredError(name='RMse', dtype=None)
from keras.layers import LeakyReLU, ReLU, Add
from tensorflow.keras.layers import Attention
adam=Adam(learning_rate=0.0001)
sgd=SGD(learning_rate=0.01, momentum=0.5)
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
l = tf.keras.regularizers.l2(1e-10)

train_X = np.load('./1000_sirius/train_X.npy')
train_y_r = np.load('./1000_sirius/train_y_r.npy')

#test_X = np.load('./1000/test_X.npy')
#test_y_r = np.load('./1000/test_y_r.npy')

val_X = np.load('./1000_sirius/val_X.npy')
val_y_r = np.load('./1000_sirius/val_y_r.npy')

# training
train_X = train_X.reshape(np.int(train_X.shape[0]/1000),1000, 1)
train_y_r = train_y_r.reshape(train_y_r.shape[0],1)
# validation
val_X = val_X.reshape(np.int(val_X.shape[0]/1000),1000, 1)
val_y_r = val_y_r.reshape(val_y_r.shape[0],1)

print("training samples :",train_X.shape[0])
print("validation samples :",val_X.shape[0])

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch_'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X   
    # First component of main path
    X = Conv1D(filters = F1, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'identity_0')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)   
    # Second component of main path (≈3 lines)
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'identity_1')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_conv_branch_'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(F1, 1, activation='relu', strides = s, name = conv_name_base + 'conv_0')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '2b')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid', name = conv_name_base + 'conv_1')(X)#, kernel_initializer = glorot_uniform(seed=0))(X)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters = F3, activation='relu', kernel_size = 1, strides = s, padding = 'valid', name = conv_name_base + 'conv_shortcut')(X_shortcut)#,
                        #kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

  
  
# Define the input as a tensor with shape input_shape
import time

for stg2_blk in [1,2,3,4]:#,9,10,11,12,13,14,15]:
    for lr in ['1e-3','1e-4']:
        for bs in [128]:
            NAME = '3_stage-no_BN({})-lr({})-bs({})-resnet'.format(stg2_blk,lr,bs, int(time.time()))
            X_input = Input((1000, 1))
            print(NAME)
            file_name = NAME+'_50ep.h5'
            file_path = './models/conv_models/resnet/'+ file_name

            # Stage 1
            X = Conv1D(2, 3, strides=1, name='conv1')(X_input)#, kernel_initializer=glorot_uniform(seed=0))(X_input)
            #X = BatchNormalization(name='bn_conv1')(X)
            X = Activation('relu')(X)

            # Stage 2
            X = convolutional_block(X, f=3, filters=[2, 2, 4], stage=2, block='a', s=1)
            for ll0 in range(stg2_blk-1):
                X = identity_block(X, 3, [2, 2, 4], stage=2, block='ab{}'.format(ll0))
                X = identity_block(X, 3, [2, 2, 4], stage=2, block='ac{}'.format(ll0))
                X = identity_block(X, 3, [2, 2, 4], stage=2, block='ad{}'.format(ll0))
                #print("added:",ll)
            # Stage 3
            X = convolutional_block(X, f=3, filters=[4, 4, 8], stage=2, block='b', s=1)
            X = identity_block(X, 3, [4, 4, 8], stage=2, block='bb')
            X = identity_block(X, 3, [4, 4, 8], stage=2, block='bc')
            for ll1 in range(stg2_blk-1):
                X = identity_block(X, 3, [4, 4, 8], stage=2, block='bd{}'.format(ll1))
            # Stage 4
            X = convolutional_block(X, f=3, filters=[8, 8, 16], stage=2, block='c', s=1)
            X = identity_block(X, 3, [8, 8, 16], stage=2, block='cb')
            X = identity_block(X, 3, [8, 8, 16], stage=2, block='cc')
            for ll2 in range(stg2_blk-1):
                X = identity_block(X, 3, [8, 8, 16], stage=2, block='cd{}'.format(ll2))

            X = MaxPooling1D(2, name='avg_pool')(X)
            #X = Conv1D(1,1,activation='relu')(X)
            # output layer
            X = Flatten()(X)
            X = Dense(1, activation='relu', name='fcc' + str(1))(X)#, kernel_initializer = glorot_uniform(seed=0))(X)

            # Create model
            model = Model(inputs = X_input, outputs = X, name='ResNet50')
            tensorboard = TensorBoard(log_dir='resnet_convblock_lr_bs_50ep/{}'.format(NAME), histogram_freq=1, profile_batch = '500, 520')
            model.compile(optimizer=Adam(learning_rate=float(lr)),loss='mse',metrics=[ 'mae', rmse])
            model.summary()
            model.fit(train_X, train_y_r,
                  batch_size=bs,
                  epochs=50,
                  verbose=1,
                  shuffle=True,
                  validation_data=(val_X, val_y_r),#)#,
                  callbacks=[tensorboard])
            model.save(file_path) 
