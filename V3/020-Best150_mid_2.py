import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
#os.environ['KERAS_BACKEND'] = 'theano'

# Tell tensorflow to only use two CPUs so I can use my computer for other stuff too.
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
# config = tf.ConfigProto(intra_op_parallelism_threads=2)
# session = tf.Session(config=config)
import keras.backend as K
# K.set_session(session)

from keras.models import Sequential, load_model, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, GaussianNoise, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam
from keras import regularizers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

#from scipy.misc import imread

np.random.seed(5)
# tf.random.set_random_seed(3) # TF 1.x

# Train/validate/test info
batch_size=int(512/8)
#NOTE: THE CLASS WEIGHTS ARE NOT EVEN
class_weight={0: 10, 1: 1}
epochs = 1000
ConvScale=32
DenseScale=64 / 4
# GN1 = .054
# GN2 = .018
# GN3 = .14
# alpha = .24
spatial_d_rate = 0.25
GN1 = 0
GN2 = 0
GN3 = 0
alpha = 0
dropout_rate = 0.3
reg_scale = 0.0001
kernel_size = 3

# Calculate the F1 score which we use for optimizing the CNN.
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

def lr_schedule(epoch):
  orig = .01
  mult = .25 
  e = 2.7
  return orig*(e**(mult*epoch))


# Load the image datasets from the HDF.
# RunDir = '/home/zack/Data/SAH/Code/Gen002/001 - CNN'
# DataDir = '/home/zack/Data/SAH/Code/Gen002/Data'

DataDir = os.path.join('..', 'Data', '10000_150x150')
DataFile = h5py.File(os.path.join(DataDir, 'Craters.hdf5'), 'r+')

# DataDir = '/home/admin/Desktop'
# DataFile = h5py.File(os.path.join(DataDir, 'Aug6','Middle_FOV150_Num10k_new.hdf5'), 'r+')
#TrainTestValSplit = DataFile.attrs['TrainTestValSplit']
FOVSize = DataFile.attrs['FOVSize']
print("FOVSize: ",  FOVSize)
NumFOVs = DataFile.attrs['NumFOVs']
try:
  Foils = DataFile.attrs['Foils'].split(',')
except:
  Foils = DataFile.attrs['Foils']
# Read the Train/Test/Val datasets.
num_ims = int(NumFOVs)
ad_sub = 0
TrainNo = np.array(DataFile['TrainNo'])
TrainYes = np.array(DataFile['TrainYes'])
TestNo = np.array(DataFile['TestNo'])
TestYes = np.array(DataFile['TestYes'])
ValNo = np.array(DataFile['ValNo'])
ValYes = np.array(DataFile['ValYes'])
print('before:', len(TrainNo))

y1 = TrainYes[20]
y2 = TrainYes[35]
n1 = TrainNo[20]
n2 = TrainNo[35]


plt.subplot(321)
plt.hist(np.ndarray.flatten(y1))
plt.title('y1')
plt.subplot(322)
plt.hist(np.ndarray.flatten(y2))
plt.title('y2')
plt.subplot(323)
plt.hist(np.ndarray.flatten(n1))
plt.title('n1')
plt.subplot(324)
plt.hist(np.ndarray.flatten(n2))
plt.title('n2')
plt.subplot(325)
plt.hist(np.ndarray.flatten(np.array(TrainYes)))
plt.title('all yes')
plt.subplot(326)
plt.hist(np.ndarray.flatten(np.array(TrainNo)))
plt.title('all no')
plt.savefig('no_side.png')
plt.close()


# TrainNo = TrainNo[:num_ims]
# print('after:', len(TrainNo))
# TrainYes = TrainYes[:num_ims]
# TestNo = TestNo[:num_ims]
# TestYes = TestYes[:num_ims]
# ValNo = ValNo[:num_ims]
# ValYes = ValYes[:num_ims]


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
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(TrainData, TrainAnswers, batch_size=batch_size, seed=3)#, save_to_dir=os.path.join(RunDir, 'Train_genimages'))
validation_generator = validation_datagen.flow(ValData, ValAnswers, batch_size=batch_size, seed=4)
test_generator = test_datagen.flow(TestData, TestAnswers, batch_size=batch_size, seed=5)

# Define the NN
# Now define the neural network.
input_shape = (FOVSize, FOVSize, 1) # Only one channel since these are B&W.

model = Sequential()
model.add(GaussianNoise(GN1, input_shape = (None, None, 1)))
model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
# model.add(GaussianNoise(GN2))
model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))

model.add(GaussianNoise(GN3))
model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
# model.add(GaussianNoise(GN3))
model.add(Conv2D(int(2*ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D())
#model.add(Dropout(dropout_rate / 2))

model.add(Conv2D(int(ConvScale), (kernel_size, kernel_size), padding='valid', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(LeakyReLU(alpha = alpha))
model.add(SpatialDropout2D(spatial_d_rate))
model.add(MaxPool2D(pool_size = 2))
model.add(Conv2D(int(ConvScale), (int(kernel_size), int(kernel_size)), padding = 'valid', activation = 'relu', kernel_regularizer = regularizers.l2(reg_scale)))
model.add(SpatialDropout2D(spatial_d_rate))

model.add(GlobalMaxPooling2D())
#model.add(Flatten())
model.add(Dense(int(2*DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(2*DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale)))
model.add(LeakyReLU(alpha = alpha))
model.add(Dropout(dropout_rate))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Nadam(lr=0.0002), loss='binary_crossentropy', metrics=['acc', f1_acc])
model.save(os.path.join(DataDir, 'NFP_FOV{}.h5'.format(FOVSize)))
model = load_model(os.path.join(DataDir, 'NFP_FOV{}.h5'.format(FOVSize)), custom_objects={'f1_acc': f1_acc})
model.summary()
# plot_model(model, to_file='NFP.png', show_shapes=True)



# Do the training
# CSVLogger is a checkpoint function.  After each epoch, it will write the stats from that epoch to a csv file.
Logger = CSVLogger(os.path.join(DataDir, 'NFP_Log_FOV{}.txt'.format(FOVSize)), append=True)
# ModelCheckpoint will save the configuration of the network after each epoch.
# save_best_only ensures that when the validation score is no longer improving, we don't overwrite
# the network with a new configuration that is overfitting.
Checkpoint1 = ModelCheckpoint(os.path.join(DataDir, 'NFP_F1_FOV{}.h5'.format(FOVSize)), verbose=1, save_best_only=True, monitor='val_f1_acc')#'val_acc')
Checkpoint2 = ModelCheckpoint(os.path.join(DataDir, 'NFP_loss_FOV{}.h5'.format(FOVSize)), verbose=1, save_best_only=True, monitor='val_loss')#'val_acc')
Checkpoint3 = ModelCheckpoint(os.path.join(DataDir, 'NFP_acc_FOV{}.h5'.format(FOVSize)), verbose=1, save_best_only=True, monitor='val_acc')#'val_acc')
EarlyStop = EarlyStopping(monitor='val_loss', patience=20)
from time import time

#TBLog = TensorBoard(log_dir = '/users/loganjaeger/Desktop/TB/testing_over_ssh/{}'.format(time()))
TBLog = TensorBoard(log_dir = os.path.join(DataDir, 'TB', '{}'.format(round(time(), 4))))
#preload_weights = load_model(os.path.join(DataDir, 'Foils_CNN_acc_FOV{}.h5'.format(FOVSize)), custom_objects={'f1_acc': f1_acc})

# If we already ran and checkpointed then we can preload previous checkpoint weights
SavedModelName = os.path.join(DataDir, 'NFP_acc_FOV{}.h5'.format(FOVSize))
if os.path.exists(SavedModelName):
    preload_weights = load_model(SavedModelName, custom_objects={'f1_acc': f1_acc})
    preload_weights = preload_weights.get_weights()
    model.set_weights(preload_weights)

model.fit_generator(generator=train_generator,
                   steps_per_epoch=train_generator.n//batch_size,
                   epochs=epochs,
                   verbose=2,
                   validation_data=validation_generator,
                   validation_steps=validation_generator.n//batch_size,
                   callbacks=[Checkpoint1, Checkpoint2, Checkpoint3, Logger, TBLog],
                   class_weight=class_weight
                   )
high_acc = load_model(os.path.join(DataDir, 'NFP_acc_FOV{}.h5'.format(FOVSize)), custom_objects={'f1_acc': f1_acc})
high_f1 = load_model(os.path.join(DataDir, 'NFP_F1_FOV{}.h5'.format(FOVSize)), custom_objects={'f1_acc': f1_acc})
low_loss = load_model(os.path.join(DataDir, 'NFP_loss_FOV{}.h5'.format(FOVSize)), custom_objects={'f1_acc': f1_acc})

#use a model MDL to predict on a file named NAME.
def calc_test_acc(name, mdl = high_acc):
  DF = h5py.File(os.path.join(DataDir, '{}.hdf5'.format(name)), 'r+')
  TestYes = DF['TestYes']
  TestNo = DF['TestNo']
  FOVSize = DF.attrs['FOVSize']
  y = mdl.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
  n = mdl.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
  cp = len([i for i in y if i > .5]) / len(y)
  cn = len([i for i in n if i < .5]) / len(n)
  print(FOVSize,' w craters:')
  print(cp)
  print(FOVSize,' no craters:')
  print(cn)
  print(FOVSize, ' total acc:')
  print((cp + cn) / 2)
  print(' ')

#calculate the accuracy on different sized FOVs
# calc_test_acc('new_to_train_200')
# calc_test_acc('new_to_train_300')
# calc_test_acc('new_to_train_450')
# calc_test_acc('new_to_train_500')

