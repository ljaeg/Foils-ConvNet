import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools
import h5py
from scipy.ndimage import measurements as meas
from scipy import ndimage as ndi
# from numba import njit, jit

def crater_do(FOVSize, PathToFile, daskclient):

    ### LOAD THE HDF.
    DataFile = h5py.File(PathToFile, 'r+')
    FOVSize = DataFile.attrs['FOVSize']
    NumFOVs = DataFile.attrs['NumFOVs']
    
    # Read the Train/Test/Val datasets.
    TrainNo = DataFile['TrainNo']
    TrainYes = DataFile['TrainYes']
    TestNo = DataFile['TestNo']
    TestYes = DataFile['TestYes']
    ValNo = DataFile['ValNo']
    ValYes = DataFile['ValYes']

    # LOAD CRATER IMAGES AND MAKE AUGMENTED IMAGES
    # The augmented images will be scaled, rotated, stretched a bit (aspect ratio).  We will add noise at the input to the CNN, so we don't do that here.
    RD = os.path.join("..", "..", "RawData")
    CraterNames = glob(pathname=os.path.join(RD, 'Alpha_Craters', '*.png'))
    Craters = []
    for c in CraterNames:
        Craters.append(imread(c)/255)
    np.random.seed(42)
    print('len of craters: ', len(Craters)) # This is just to make sure we've actually got some craters

    # TrainYesArr = np.zeros(TrainYes.shape)
    # TrainYes.read_direct(TrainYesArr)
    # ValYesArr = np.zeros(ValYes.shape)
    # ValYes.read_direct(ValYesArr)
    # TestYesArr = np.zeros(TestYes.shape)
    # TestYes.read_direct(TestYesArr)

    print('Creating training craters.')
    AddCraters(TrainYes, Craters, FOVSize[0], daskclient)
    DataFile.flush()
    print('Creating test craters.')
    AddCraters(TestYes, Craters, FOVSize[0], daskclient)
    DataFile.flush()
    print('Creating validation craters.')
    AddCraters(ValYes, Craters, FOVSize[0], daskclient)
    DataFile.flush()

    ### CLEANUP
    DataFile.close()

    print('Done.')

def AddCraters(Data, Craters, FOVSize, daskclient):
    # We want to randomize the properies of the augmented images.  All the transformation parameters are uniformly distributed except aspect ratio which should hew close to 1 so we use Gaussian.
    scale = np.random.uniform(low = 0, high = 30 / FOVSize, size = Data.shape[0])
    rotate = np.random.random(Data.shape[0])*360
    shift = np.random.uniform(low = .1, high = .9, size = (Data.shape[0],2)) - .5
    aspect = np.random.normal(1, 0.1, Data.shape[0])
    CraterIndices = np.random.randint(len(Craters), size=Data.shape[0])

    scatterCraters = daskclient.scatter(Craters)

    # Now make all the transformed craters
    Futures = []
    for n, i in enumerate(CraterIndices):
        if n % 300 == 0:
            print(f'Submitting {n}')
        Futures.append(daskclient.submit(DoOneCrater, n, i, scatterCraters, scale[n], rotate[n], shift[n,:], aspect[n], FOVSize))

    DataArr = np.zeros(Data.shape)
    Data.read_direct(DataArr)

    for f in Futures:
        n,g,a = f.result()
        if n % 100 == 0:
            print(f'Receiving {n}')
        # Merge the transformed crater with the plain FOV.
        DataArr[n] = DataArr[n]*(1-a) + g
        del f # Important to garbage collect these as we go or we'll run out of memory holding all these in RAM.

    Data.write_direct(DataArr)

def DoOneCrater(n, i, Craters, scale, rotate, shift, aspect, FOVSize):
    grayscale = Craters[i][:,:,0]
    alpha = Craters[i][:,:,3]
    g, a = ImageTools.TransformImage(grayscale, alpha, scale=scale, rotate=rotate, shift=shift, aspect=aspect, FOVSize=FOVSize)
    return n,g,a

