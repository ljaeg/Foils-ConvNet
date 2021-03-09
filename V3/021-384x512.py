"""
This is where we do transfer learning, taking a network trained on 150x150 images and doing a few epochs on 384x512 images.
A few things of note:
-If you want to train the network, you must change SHOULD_FIT to True. 
    >Because the network has already trained, this program is set up so that you'll just see the results of predicting on the test set.
    >If you train, I recommend changing the save files so it doesn't overwrite the earlier network.
-The hdf5 files I'm currently loading in, YES_TRAIN and NO_TRAIN, are calibration images consistently marked as positive by dusters and 
real images consistently marked as negative by dusters, respectively.
-the class weighting is currently adjusted to bias ~against~ false positives.
"""

import numpy as np
import pandas as pd
import os
import h5py
from time import time

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Let tensorflow use more memory on the GPUs.
session = tf.Session(config=config) 
import keras.backend as K

from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam
from keras import regularizers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

np.random.seed(5)
tf.random.set_random_seed(3)

batch_size = int(8)
class_weight ={0: 10, 1: 1}

#F1 score
def f1_acc(y_true, y_pred):

    # import numpy as np

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    # if K.isnan(f1_score):
    #     print('c1:', c1)
    #     print('c2:', c2)
    #     print('c3:', c3)

    return f1_score

#Load in the 384x512 data to train on
DataDir = os.path.join('..', 'Data', '1000_384x512')
Data = h5py.File(os.path.join(DataDir, 'Craters.hdf5'), 'r')
# DataYes = h5py.File(os.path.join(DataDir, 'Dataamazon2k_hascraters.hdf5'), 'r')
# DataNo = h5py.File(os.path.join(DataDir, 'Dataamazon2k_nocraters.hdf5'), 'r')

# Generally the training set is too big and we have to truncate it at some number where it still fits in the GPU.
MaxNumImages = 500

TrainYes = Data["TrainYes"][0:MaxNumImages,...]
TrainNo = Data["TrainNo"][0:MaxNumImages,...]
TestYes = Data["TestYes"][0:MaxNumImages,...]
TestNo = Data["TestNo"][0:MaxNumImages,...]
ValYes = Data["ValYes"][0:MaxNumImages,...]
ValNo = Data["ValNo"][0:MaxNumImages,...]

# Concatenate the no,yes crater chunks together to make cohesive training sets.
TrainData = np.concatenate((TrainNo,TrainYes), axis=0)[:,:,:,np.newaxis]
TestData = np.concatenate((TestNo,TestYes), axis=0)[:,:,:,np.newaxis]
ValData = np.concatenate((ValNo,ValYes), axis=0)[:,:,:,np.newaxis]


# And make answer vectors
TrainAnswers = np.ones(len(TrainNo) + len(TrainYes))
TrainAnswers[:len(TrainNo)] = 0
TestAnswers = np.ones(len(TestNo) + len(TestYes))
TestAnswers[:len(TestNo)] = 0
ValAnswers = np.ones(len(ValNo) + len(ValYes))
ValAnswers[:len(ValNo)] = 0

# Make generators to stream them.
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
#test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)
#test_generator = test_datagen.flow(TestData, TestAnswers, batch_size=batch_size, seed=5)

#load the model
model = load_model(os.path.join(DataDir, 'NFP_acc_FOV[{} {}].h5'.format(150, 150)), custom_objects={'f1_acc': f1_acc})

Checkpoint1 = ModelCheckpoint(os.path.join(DataDir, 'NFP_actual_f1.h5'), verbose=1, save_best_only=True, monitor='val_f1_acc')
Checkpoint2 = ModelCheckpoint(os.path.join(DataDir, 'NFP_actual_loss.h5'), verbose=1, save_best_only=True, monitor='val_loss')
Checkpoint3 = ModelCheckpoint(os.path.join(DataDir, 'NFP_actual_acc.h5'), verbose=1, save_best_only=True, monitor='val_acc')
TBLog = TensorBoard(log_dir = os.path.join(DataDir, 'TB', '{}'.format(round(time(), 4))))

#Fit the model, if we want to.
should_fit = True # if false, we want to just load in the model and see the predictions on test data. If true, we want to fit the model on our train generator.
if should_fit:
    model.fit_generator(generator=train_generator,
                       steps_per_epoch=train_generator.n//batch_size,
                       epochs=300,
                       verbose=2,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.n//batch_size,
                       callbacks=[Checkpoint1, Checkpoint2, Checkpoint3, TBLog],
                       class_weight=class_weight
                       )

high_acc = load_model(os.path.join(DataDir, 'NFP_actual_acc.h5'), custom_objects={'f1_acc': f1_acc})
high_f1 = load_model(os.path.join(DataDir, 'NFP_actual_f1.h5'), custom_objects={'f1_acc': f1_acc})
low_loss = load_model(os.path.join(DataDir, 'NFP_actual_loss.h5'), custom_objects={'f1_acc': f1_acc})


FOV1 = 384
FOV2 = 512

print(TestNo.shape)
no_preds = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOV1, FOV2, 1)))
yes_preds = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOV1, FOV2, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("high acc:")
print("no: ", x)
print("yes: ", y)
print(' ')

no_preds = high_f1.predict(np.reshape(TestNo, (len(TestNo), FOV1, FOV2, 1)))
yes_preds = high_f1.predict(np.reshape(TestYes, (len(TestYes), FOV1, FOV2, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("high f1:")
print("no: ", x)
print("yes: ", y)
print(' ')

no_preds = low_loss.predict(np.reshape(TestNo, (len(TestNo), FOV1, FOV2, 1)))
yes_preds = low_loss.predict(np.reshape(TestYes, (len(TestYes), FOV1, FOV2, 1)))
x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)
print("low loss:")
print("no: ", x)
print("yes: ", y)
