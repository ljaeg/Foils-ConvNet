"""
Fully connected pipeline. Specify FOVSize and NumImgs, and this will make a hdf5 file with the datasets
TrainYes, TrainNo, TestYes, TestNo, ValYes, and ValNo, each with the number of images equal to NumImgs,
and the image size FOVSize x FOVSize x 1 (grayscale).
The datasets TrainYes, TestYes, and ValYes will all contain craters.
All of the datasets will be normalized in one of two ways (or both ways), which can be changed in the make_norm.py file.
The hdf5 file will be saved under the path SavePath.

Then, another training set will be created using Full images (384x512 pixels) and normalized in the same way with Yes/No, Train/Val,Test splits.

Finally a third set can be made with just random images -- not for training but for predicting and finding real craters.
"""

import make_norm
import os
from dask.distributed import Client
from datetime import datetime
from rich.traceback import install; install()
import h5py
import numpy as np
from numba import njit

@njit()
def ArrayFromShuffledIndexes(Source, Shape, Indices):
    Dest  =  np.zeros(Shape)
    for j, i in enumerate(Indices):
        Dest[j,...] = Source[i,...]
    return Dest

def SplitTVT(SplitRatios, Data):
    # Verify that the shape of the input data is correct.  It should be N_X_Y, N is number of images, X and Y are pixels.
    assert len(Data.shape) == 3, "Input data must be three dimensional: Number of images, and X,Y pixels."
    # Generate a randomized indexation into the data.
    DataIndices = np.array(range(Data.shape[0]))
    np.random.shuffle(DataIndices)
    # Make three Indices.
    SplitPoints = np.cumsum(np.array(SplitRatios)*len(DataIndices)).astype(int)
    IndTrain  =  DataIndices[:SplitPoints[0]]
    IndVal    =  DataIndices[SplitPoints[0]:SplitPoints[1]]
    IndTest   =  DataIndices[SplitPoints[1]:SplitPoints[2]]
    # Make three numpy arrays from the random indexes
    DataArr = np.zeros(Data.shape)
    Data.read_direct(DataArr)
    Train  =  ArrayFromShuffledIndexes(DataArr,  (len(IndTrain),  Data.shape[1],  Data.shape[2]),  IndTrain)
    Val    =  ArrayFromShuffledIndexes(DataArr,  (len(IndVal),    Data.shape[1],  Data.shape[2]),  IndVal)
    Test   =  ArrayFromShuffledIndexes(DataArr,  (len(IndTest),   Data.shape[1],  Data.shape[2]),  IndTest)
    return Train, Val, Test

# Combine two HDFs into one and do Train/Val/Test splits.
def CombineHDFsIntoTVT(SplitRatios, InputYesFileName, InputNoFileName, OutputFileName):
    # This file will contain the Yes/No x Train/Val/Test sets (6 total).
    OutputData = h5py.File(OutputFileName, 'w')
    # Open the Yes data.
    InputData = h5py.File(InputYesFileName, 'r')
    # Split it and dump it.
    print("Splitting Yes set into Train/Val/Test.")
    TrainYes, ValYes, TestYes = SplitTVT(SplitRatios, InputData['images'])
    print("Writing TrainYes")
    OutputData.create_dataset("TrainYes", TrainYes.shape, data=TrainYes, chunks=True, compression='gzip', compression_opts=2)
    print("Writing ValYes")
    OutputData.create_dataset("ValYes", ValYes.shape, data=ValYes, chunks=True, compression='gzip', compression_opts=2)
    print("Writing TestYes")
    OutputData.create_dataset("TestYes", TestYes.shape, data=TestYes, chunks=True, compression='gzip', compression_opts=2)
    # And close the Yes inputs.
    InputData.close()
    del TrainYes, ValYes, TestYes
    # Now open the No data.
    InputData = h5py.File(InputNoFileName, 'r')
    # Split it and dump it.
    TrainNo, ValNo, TestNo = SplitTVT(SplitRatios, InputData['images'])
    print("Writing TrainNo")
    OutputData.create_dataset("TrainNo", TrainNo.shape, data=TrainNo, chunks=True, compression='gzip', compression_opts=2)
    print("Writing ValNo")
    OutputData.create_dataset("ValNo", ValNo.shape, data=ValNo, chunks=True, compression='gzip', compression_opts=2)
    print("Writing TrainNo")
    OutputData.create_dataset("TestNo", TestNo.shape, data=TestNo, chunks=True, compression='gzip', compression_opts=2)
    # And close the No inputs.
    InputData.close()
    del TrainNo, ValNo, TestNo
    # Write the attributes that are expected.
    OutputData.attrs['NumFOVs'] = OutputData['TrainYes'].shape[0]
    OutputData.attrs['FOVSize'] = OutputData['TrainYes'].shape[1:]
    OutputData.attrs['Foils'] = 'Unspecified foil number(s).'
    # Close the output file
    OutputData.close()
    return


if __name__ == '__main__':
    t0 = datetime.now()
    # client = Client(processes=True)

    FOVSize = (384,512)
    Dir = os.path.join("..", "Data")
    OutputDir = os.path.join(Dir, '1000_384x512')
    OutputFileName = os.path.join(OutputDir, 'Craters.hdf5')
    InputYesFileName = os.path.join(Dir, 'amazon2k_FullFOV_yes.hdf5')
    InputNoFileName = os.path.join(Dir, 'amazon2k_FullFOV_no.hdf5')
    SplitRatios = [1/3., 1/3., 1/3.]

    if not os.path.exists(OutputDir):
        os.mkdir(OutputDir)

    print(f'Input Yes File: {InputYesFileName}')
    print(f'Input No File: {InputNoFileName}')
    print(f'Output Combined File: {OutputFileName}')
    # Combine input HDFs and produce an output HDF.
    CombineHDFsIntoTVT(SplitRatios, InputYesFileName, InputNoFileName, OutputFileName)
    
    ### Normalize the images
    make_norm.norm_do(OutputFileName, norm_type = "both")
    print('normed \n')

    print('!!!!!!!!!!!!!!!!!!!')
    print(f'Time to run: {datetime.now() - t0}')
    print('all done')
    print('!!!!!!!!!!!!!!!!!!!')
