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
from make_norm import both

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(tf.compat.v1.logging.FATAL)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True #Let tensorflow use more memory on the GPUs.
session = tf.compat.v1.Session(config=config) 
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
from imageio import imread
from glob2 import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--MaxFiles', default=-1)
parser.add_argument('--PredictDir', default=os.path.join('..', 'Data', 'Predict', 'amazonimages'))
parser.add_argument('--PredictionsOutputFile', default=os.path.join('..', 'Data', 'Predict', 'amazonimages.txt'))
parser.add_argument('--Extension', default='jpg')
parser.add_argument('--ModelFile', default=os.path.join('..', 'Data', '1000_384x512', 'NFP_actual_acc.h5'))
parser.add_argument('--Verbose', default=True)
args = parser.parse_args()

np.random.seed(5)
tf.compat.v1.random.set_random_seed(3)

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

#load the model
model = load_model(args.ModelFile, custom_objects={'f1_acc': f1_acc})

# Clear the output file.  From here on we append.
with open(args.PredictionsOutputFile, 'w') as f:
    f.write('Prediction,FileName\n')

# Get all the files we are supposed to analyze.
Files = glob(os.path.join(args.PredictDir, '*.'+args.Extension))
if int(args.MaxFiles) < 0:
    NumFiles = len(Files)
else:
    NumFiles = int(args.MaxFiles)
print(f'Predicting {NumFiles} images from {args.PredictDir}.')

# Loop through all the files and predict them.
for i, FileName in enumerate(Files[:NumFiles]):
    try:
        PredictData = imread(FileName)[np.newaxis, ..., np.newaxis]
        PredictData = both(PredictData)
        Predictions = model.predict(PredictData)
        OutStr = f'{Predictions[0][0]},{FileName}'
        if args.Verbose == True:
            print(OutStr)
        else:
            print('.', end='', flush=True)
        with open(args.PredictionsOutputFile, 'a') as f:
            f.write(OutStr + '\n')
    except Exception as e:
        print()
        print(f'Error encountered while attempting to predict: {FileName}')
        print(e)

print()
print(f'Predictions written to: {args.PredictionsOutputFile}.')
print('Done!')
        
