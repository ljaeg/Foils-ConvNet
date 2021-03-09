import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import os, shutil
from glob2 import glob
from imageio import imread
import ImageTools
import h5py

def GetGlobbedFiles(InputFileName, Foils, RawDir):
    ### SCAN THE RAW DATA
    # We can accept a list of file names in text format and we will produce an output HDF from that.
    if os.path.splitext(InputFileName)[-1] == '.txt':
        # We don't need to redo globbing if we already globbed.
        try:
            with open(InputFileName, 'r') as f:
                GlobbedFiles = f.read().splitlines()
        except IOError as e:
            GlobbedFiles = []
            for d in Foils:
                d = os.path.join(RawDir, d, '*.tif')
                g = glob(pathname=d)
                GlobbedFiles += list(g)
            with open(InputFileName, 'w') as f:
                f.writelines('%s\n' % n for n in GlobbedFiles)
        print('There are %d image files in the raw data.' % len(GlobbedFiles))

    else:
        # Otherwise we will pull the images out of an existing HDF file under the images group.
        InputFile = h5py.File(InputFileName, 'r')
        ImagesData = InputFile['images']
        GlobbedFiles = list(range(ImagesData.shape[0]))
        InputFile.close()
    return GlobbedFiles

def blanks_do(FOVSize, FullFOV, NumFOVs, DataDir, InputYesFileName, InputNoFileName, OutputFileName, daskclient):
    ### SETUP PARAMETERS
    # Raw data is stored locally.
    RawDir = os.path.join("..", "..", "RawData") 
    Foils = ['I1009N', 'I1126N', 'I1126N_2', 'I1126N_3']
    TrainTestValSplit = [.33, .33, .33] #currently this hace no bearing on anything except the attribute in the hdf5 file
    
    NumTrain = NumFOVs
    NumTest = NumFOVs
    NumVal = NumFOVs
    
    GlobbedFilesYes = GetGlobbedFiles(InputYesFileName, Foils, RawDir)
    GlobbedFilesNo = GetGlobbedFiles(InputYesFileName, Foils, RawDir)

    ### MAKE HDF TO HOLD OUR IMAGES.
    DataFile = h5py.File(OutputFileName, 'w')
    DataFile.attrs['TrainTestValSplit'] = TrainTestValSplit
    DataFile.attrs['FOVSize'] = FOVSize
    DataFile.attrs['NumFOVs'] = NumFOVs
    DataFile.attrs['Foils'] = ', '.join(Foils)

    # Yes means this image has a crater.
    # No means there is no crater.
    compression_opts = 2
    TrainYes = DataFile.create_dataset('TrainYes', (NumTrain, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)
    TrainNo  = DataFile.create_dataset('TrainNo',  (NumTrain, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)

    TestYes  = DataFile.create_dataset('TestYes', (NumTest, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)
    TestNo   = DataFile.create_dataset('TestNo',  (NumTest, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)

    ValYes   = DataFile.create_dataset('ValYes', (NumVal, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)
    ValNo    = DataFile.create_dataset('ValNo',  (NumVal, FOVSize[0], FOVSize[1]), dtype='f8', chunks=(1,FOVSize[0], FOVSize[1]), compression='gzip', compression_opts=compression_opts)

    DataFile.flush()

    ### POPULATE TRAIN/TEST/VAL WITH NO CRATER IMAGES
    # Choose random files from which to draw FOVs.
    np.random.seed(5)

    if FullFOV == True:
        # Grab all the FOVs we want.
        GetFullFOVs(GlobbedFilesNo, TrainNo, daskclient, InputNoFileName)
        DataFile.flush()
        GetFullFOVs(GlobbedFilesYes, TrainYes, daskclient, InputYesFileName)
        DataFile.flush()
        GetFullFOVs(GlobbedFilesNo, TestNo, daskclient, InputNoFileName)
        DataFile.flush()
        GetFullFOVs(GlobbedFilesYes, TestYes, daskclient, InputYesFileName)
        DataFile.flush()
        GetFullFOVs(GlobbedFilesNo, ValNo, daskclient, InputNoFileName)
        DataFile.flush()
        GetFullFOVs(GlobbedFilesYes, ValYes, daskclient, InputYesFileName)
        DataFile.flush()
    else:
        # Grab all the FOVs we want.
        GetRandomFOVs(GlobbedFilesNo, TrainNo, daskclient)
        DataFile.flush()
        GetRandomFOVs(GlobbedFilesYes, TrainYes, daskclient)
        DataFile.flush()
        GetRandomFOVs(GlobbedFilesNo, TestNo, daskclient)
        DataFile.flush()
        GetRandomFOVs(GlobbedFilesYes, TestYes, daskclient)
        DataFile.flush()
        GetRandomFOVs(GlobbedFilesNo, ValNo, daskclient)
        DataFile.flush()
        GetRandomFOVs(GlobbedFilesYes, ValYes, daskclient)
        DataFile.flush()

    ### CLEANUP
    DataFile.close()

def GetOneFOV(FileName, FOVSize, i):
    img = imread(FileName)
    FOV = ImageTools.GetRandomFOV(img, FOVSize)
    return i, FOV

def GetRandomFOVs(GlobbedFiles, Data, daskclient):
    # It is slow to read each image.  So we will take SpeedupFactor FOVs from each in order to speed up I/O.
    NumFOVs = Data.shape[0] # The number of images to get is the first axis of the data cube the caller wants filled.
    FOVSize = Data.shape[1] # x for each image.
    assert(Data.shape[1] == Data.shape[2]) # x == y for each image.

    DataArr = np.zeros(Data.shape)
    Data.read_direct(DataArr)

    # Quick routine for waiting on a batch of futures so we don't run out of memory.
    def CleanUpFutures(Futures, Data, i):
        for f in Futures:
            if (i % 100) == 0:
                print('%s: #%d'%(Data.name, i))
            i, FOV = f.result()
            # Pull a FOV out at random and put it in the HDF.
            DataArr[i,:,:] = FOV/255.0
            del f

    Files = np.random.choice(GlobbedFiles, int(NumFOVs)) 
    Futures = []
    for i, FileName in enumerate(Files):
        Futures.append(daskclient.submit(GetOneFOV, FileName, FOVSize, i))
        if (i > 0) and ((i % 1000) == 0):
            CleanUpFutures(Futures, Data, i)
            Futures = []
    CleanUpFutures(Futures, Data, i)
    Futures = []
    print('%s: #%d'%(Data.name, i))
    Data.write_direct(DataArr)

def GetFullFOVs(GlobbedFiles, Data, daskclient, InputFileName):
    InputFile = h5py.File(InputFileName, 'r')
    ImagesData = InputFile['images']

    NumFOVs = Data.shape[0] # The number of images to get is the first axis of the data cube the caller wants filled.

    DataArr = np.zeros(Data.shape)
    Data.read_direct(DataArr)

    ImageIndexes = np.random.choice(GlobbedFiles, int(NumFOVs))
    for i in len(ImageIndexes):
        FOV = ImagesData[ImageIndexes[i],:,:]
        DataArr[i,:,:] = FOV/255.0

    Data.write_direct(DataArr)
